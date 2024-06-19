# import cv2
import os
from argparse import ArgumentParser
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from time import time
import open_clip
import torch
import pickle
import numpy as np
import math
import json
from transformers import AutoImageProcessor, AutoModel


from utils.utils import CustomImageFolder, WatermarkDataset, calc_match_rate


from python_polar_coding.polar_codes import FastSSCPolarCodec, SCPolarCodec


# inplace function
# compare similarity score to max similarity scores, and insert if it's high enough
# max_sims is sorted in asceding order
def insert_sim(sim, id, max_sims, max_sim_ids):
    for i in range(len(max_sims)):
        if sim <= max_sims[i]:
            if i > 0:
                max_sims[i-1] = sim
                max_sim_ids[i-1] = id
            return
        # shift to the left
        if i > 0:
            max_sims[i-1] = max_sims[i]
            max_sim_ids[i-1] = max_sim_ids[i]

    # sim has the highest value among max_sims
    max_sims[-1] = sim
    max_sim_ids[-1] = id


def get_codec_confidence(codec):
    llrs = codec.decoder.intermediate_llr[-1]
    if(len(llrs) > 1):
        print("ERORRRRRRRRRR")
        exit()
    confidence_score = abs(llrs[0]) / codec.N

    return confidence_score


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", default="clip-openai", type=str,
        choices=["clip-openai", "dinov2"])
    parser.add_argument("--wm-method", default="trustmark", type=str,
        choices=["trustmark"]) # choices=["stegaStamp", "trustmark"])  # StegaStamp currently unavailable
    parser.add_argument("--query-dir", type=str, required=True)
    parser.add_argument("--emb-path", type=str, required=True)
    parser.add_argument("--cluster-info-path", type=str, required=True)
    parser.add_argument("--wm-th", type=float, default=0.6, help="threshold for watermark cluster match")
    parser.add_argument("--log-path", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=1, help="evaluate method on top-k accuracy")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model == "clip-openai":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
        model.to(device)
        model.eval()
        def model_fn(image):
            emb = model.encode_image(preprocess(image).unsqueeze(0).to(device)).squeeze(0)
            emb /= emb.norm()
            return emb
        
    elif args.model == "dinov2":
        processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        # loader_process = lambda x: (processor_dino(images=x, return_tensors="pt").to(device))["pixel_values"]
        model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
        def model_fn(image):
            inputs = processor_dino(images=image, return_tensors="pt").to(device)
            outputs = model_dino(**inputs)
            image_features_dino = outputs.last_hidden_state
            emb = image_features_dino.mean(dim=1)
            emb /= emb.norm(dim=1, keepdim=True)
            return emb.squeeze(0)
        
    else:
        raise ModuleNotFoundError(f"Model {args.model} not implemented")

    log_file = open(args.log_path, 'a')
    log_file.write("----------------------------------\n")
    log_file.write(json.dumps(vars(args), indent=2) + "\n")
    log_file.write("----------------------------------\n")

    if args.wm_method == "stegaStamp":
        from utils.stega_stamp import StegaStampWatermark
        wm_method = StegaStampWatermark()
    elif args.wm_method == "trustmark":
        from utils.trustmark import Trustmark
        wm_method = Trustmark(wm_bits=100)
    else:
        raise ModuleNotFoundError(f"Method {args.wm_method} not implemented")
    

    wm_db = WatermarkDataset(dataset=None)
    wm_db.load_cluster_info(args.cluster_info_path)

    with open(args.emb_path, 'rb') as handle:
        emb_dict = pickle.load(handle)

    query_dataset = CustomImageFolder(args.query_dir, return_id=True)

    
    N = wm_db.n_bits
    K = int(math.log2(wm_db.n_clusters))

    if N == 100:
        K_64 = math.ceil(K * 2 / 3)
        K_32 = K - K_64

        # codec_64 = FastSSCPolarCodec(N=64, K=K_64, design_snr=0.0)
        # codec_32 = FastSSCPolarCodec(N=32, K=K_32, design_snr=0.0)
        codec_64 = SCPolarCodec(N=64, K=K_64, design_snr=0.0)
        codec_32 = SCPolarCodec(N=32, K=K_32, design_snr=0.0)
    else:
        codec = SCPolarCodec(N=N, K=K, design_snr=0.0)


    with torch.no_grad():
        emb_sim_cnt = 0
        accuracy = 0.
        wm_accuracy = 0.
        avg_bit_flip = 0.
        skip_count = 0.
        avg_bit_flip_no_skip = 0.
        no_skip_acc = 0.

        for query_image, _, query_img_id in tqdm(query_dataset, total=len(query_dataset)):
            query_emb = model_fn(query_image).cpu().numpy()
            key = wm_method.decode_single_image(query_image)

            if N == 100:
                key_64 = key[:64]
                key_32 = key[64:64+32]
                
                key_64_np = []
                for x in key_64:
                    key_64_np.append(int(x))
                key_64_np = -2 * np.float32(key_64_np) + 1

                key_32_np = []
                for x in key_32:
                    key_32_np.append(int(x))
                key_32_np = -2 * np.float32(key_32_np) + 1

                cluster_id_64 = codec_64.decode(key_64_np)
                cluster_id_32 = codec_32.decode(key_32_np)
                query_cl_id = list(cluster_id_64) + list(cluster_id_32)
                query_cl_id = "".join([str(int(x)) for x in query_cl_id])
                
                conf_64 = get_codec_confidence(codec_64)
                conf_32 = get_codec_confidence(codec_32)

                wm_th = args.wm_th
                skipped = False
                if conf_64 < 0.5 or conf_32 < 0.5:
                    wm_th = 0.0
                    skipped = True
                    skip_count += 1

            else:
                key_np = []
                for x in key:
                    key_np.append(int(x))
                key_np = -2 * np.float32(key_np) + 1
                query_cl_id = codec.decode(key_np)
                query_cl_id = "".join([str(int(x)) for x in query_cl_id])
                conf = get_codec_confidence(codec)

                wm_th = args.wm_th
                skipped = False
                if conf < 0.5:
                    wm_th = 0.0
                    skipped = True
                    skip_count += 1


            gt_cluster = -1
            match_list = []
            for cl_id, img_ids in wm_db.cluster_id_to_img_id.items(): # iterate over binary codes of clusters
                match_r = calc_match_rate(cl_id, query_cl_id) # find matching rate of codes to the query's key
                match_list.append((match_r, cl_id))
                if query_img_id in img_ids:
                    gt_cluster = cl_id
            

            match_list.sort()

            # perform embedding based retrieval on matched clusters
            max_sims = [-1 for _ in range(args.top_k)]
            max_sim_ids = [-1 for _ in range(args.top_k)]
            img_in_clusters = False
            for match_r, cl_id in match_list:
                if match_r < wm_th:
                    continue
                for img_id in wm_db.cluster_id_to_img_id[cl_id]:
                    if query_img_id == img_id:
                        img_in_clusters = True
                    if not img_id in emb_dict:
                        print("ERROR", img_id)
                        continue
                    emb = emb_dict[img_id]
                    sim = np.dot(emb, query_emb)
                    emb_sim_cnt += 1
                    insert_sim(sim, img_id, max_sims, max_sim_ids)
            
            accuracy += (query_img_id in max_sim_ids) / len(query_dataset)
            avg_bit_flip += (1 - calc_match_rate(gt_cluster, query_cl_id)) / len(query_dataset)
            if not skipped:
                avg_bit_flip_no_skip += (1 - calc_match_rate(gt_cluster, query_cl_id))
                wm_accuracy += (img_in_clusters)
                no_skip_acc += (query_img_id in max_sim_ids)
    
    avg_bit_flip_no_skip /= max(1, len(query_dataset) - skip_count)
    wm_accuracy /= max(1, len(query_dataset) - skip_count)
    no_skip_acc /= max(1, len(query_dataset) - skip_count)

    log_file.write("Method Total Accuracy: {:.4f}\n".format(accuracy))
    log_file.write("No Skip Accuracy: {:.4f}\n".format(no_skip_acc))
    log_file.write("Watermark Decoder No Skip Accuracy: {:.4f}\n".format(wm_accuracy))
    log_file.write("Average Watermark Bit Flip: {:.4f}\n".format(avg_bit_flip))
    log_file.write("Average Watermark No Skip Bit Flip: {:.4f}\n".format(avg_bit_flip_no_skip))
    log_file.write("Embedding Similarity Check Ratio: {:.4f}\n".format(emb_sim_cnt / (len(query_dataset) * len(emb_dict))))
    log_file.write("Skipped Watermarks: {:.4f}\n".format((skip_count / len(query_dataset))))
    log_file.write("----------------------------------\n")

    

    with torch.no_grad():
        emb_sim_cnt = 0
        accuracy = 0.

        for query_image, _, query_img_id in tqdm(query_dataset, total=len(query_dataset)):
            query_emb = model_fn(query_image).cpu().numpy()

            max_sims = [-1 for _ in range(args.top_k)]
            max_sim_ids = [-1 for _ in range(args.top_k)]

            for img_id, emb in emb_dict.items():
                sim = np.dot(emb, query_emb) 
                emb_sim_cnt += 1
                insert_sim(sim, img_id, max_sims, max_sim_ids)
            
            accuracy += (query_img_id in max_sim_ids) / len(query_dataset)
    
    log_file.write("Naive Accuracy: {:.4f}\n".format(accuracy))
    log_file.write("----------------------------------\n")
    log_file.write("\n\n")

    
    log_file.close()
    

    
    


if __name__ == "__main__":
    main()
