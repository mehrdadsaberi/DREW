import os
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from time import time
import torch
from torchvision.utils import save_image

from utils.utils import CustomImageFolder, WatermarkDataset, CustomDataLoader


def main():
    parser = ArgumentParser()
    parser.add_argument("--wm-method", default="trustmark", type=str,
        choices=["trustmark"]) # choices=["stegaStamp", "trustmark"]) # StegaStamp currently unavailable
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--n-bits", default=100, type=int)
    parser.add_argument("--n-clusters", type=int)
    parser.add_argument("--data-cnt", type=int, default=-1)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    assert os.path.exists(args.data_dir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.wm_method == "stegaStamp":
        from utils.stega_stamp import StegaStampWatermark
        wm_method = StegaStampWatermark()
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
            ]
        )
    elif args.wm_method == "trustmark":
        from utils.trustmark import Trustmark
        wm_method = Trustmark(wm_bits=args.n_bits)
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
            ]
        )
    else:
        raise ModuleNotFoundError(f"Method {args.wm_method} not implemented")
    

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform, return_id=True,
                                data_cnt=args.data_cnt, shuffle=True)

    wm_db = WatermarkDataset(dataset)
    print("Creating binary codes...")
    if args.n_bits == 100:
        print("MODE: ecc_polar_100")
        wm_db.create_clusters(args.n_bits, args.n_clusters, mode="ecc_polar_100")
    elif (args.n_bits & (args.n_bits-1) == 0) and args.n_bits != 0: # check if n_bits is power of 2
        print("MODE: ecc_polar")
        wm_db.create_clusters(args.n_bits, args.n_clusters, mode="ecc_polar")
    else:
        print("MODE: random")
        wm_db.create_clusters(args.n_bits, args.n_clusters, mode="random")



    print(f"Finished. Loading took {time() - s:.2f}s")
    print(f"Dataset Size: {len(dataset)}")
    
    print(f"Watermarking images with {args.wm_method} method...")

    output_image_dir = os.path.join(args.out_dir, "images")
    metadata_dir = os.path.join(args.out_dir, "cluster_info.pkl")

    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    

    wm_db.store_cluster_info(metadata_dir)
    
    dataloader = CustomDataLoader(dataset, batch_size=32, shuffle=False)

    file_set = set(list(os.listdir(output_image_dir)))
    print("set created")

    with torch.no_grad():
        for images, _, img_ids in tqdm(dataloader, total=len(dataloader)):

            keys = [wm_db.cluster_id_to_code[wm_db.img_id_to_cluster_id[img_id]] for img_id in img_ids]
            wm_images = wm_method.watermark_batch_images(images, keys)

            for wm_image, img_id in zip(wm_images, img_ids):
                wm_image.save(os.path.join(output_image_dir, f"{img_id}.png"))

    print(f"Watermarking finished!")



if __name__ == "__main__":
    main()
