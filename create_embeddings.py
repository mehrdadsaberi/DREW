import cv2
import os
from argparse import ArgumentParser
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from time import time
import open_clip
import torch
import pickle
from transformers import AutoImageProcessor, AutoModel

from utils.utils import CustomImageFolder, WatermarkDataset, CustomDataLoader


class ImageCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model
        
    def forward(self,image):
        return self.model.encode_image(image)

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", default="clip-openai", type=str,
        choices=["clip-openai", "dinov2"])
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    args = parser.parse_args()

    assert os.path.exists(args.data_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model == "clip-openai":
        clip_model, _, transform = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
        loader_process = lambda x: torch.stack(x, dim=0)
        clip_image_model = ImageCLIP(clip_model)
        clip_image_model = torch.nn.DataParallel(clip_image_model)
        clip_image_model.to(device)
        clip_image_model.eval()
        def model_fn(images):
            emb = clip_image_model(images.to(device))
            emb /= emb.norm(dim=1, keepdim=True)
            return emb
        
    elif args.model == "dinov2":
        processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        transform = lambda x: x
        loader_process = lambda x: (processor_dino(images=x, return_tensors="pt").to(device))["pixel_values"]
        model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
        def model_fn(images):
            outputs_dino = model_dino(images)
            image_features_dino = outputs_dino.last_hidden_state
            emb = image_features_dino.mean(dim=1)
            emb /= emb.norm(dim=1, keepdim=True)
            return emb
        
    else:
        raise ModuleNotFoundError(f"Model {args.model} not implemented")

    dataset = CustomImageFolder(args.data_dir, transform=transform,
        return_id=True, fname_format="*.png")
    dataloader = CustomDataLoader(dataset, batch_size=64, shuffle=False)
    print(f"Finished loading {len(dataset)} data!")

    emb_dict = {}

    with torch.no_grad():
        for images, _, img_ids in tqdm(dataloader, total=len(dataloader)):
            images = loader_process(images)
            emb = model_fn(images).cpu().numpy()
            for i in range(len(img_ids)):
                emb_dict[img_ids[i]] = emb[i]
    
    with open(args.out_path, 'wb') as handle:
        pickle.dump(emb_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
