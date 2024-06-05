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
import shutil

from utils.utils import CustomImageFolder, DiffPure


def get_transforms(aug_str):
    aug = aug_str.split(',')[0]
    params = aug_str.split(',')[1:]
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            # transforms.ToTensor(),
        ]
    )
    org_transform = transform

    if aug == 'no_aug':
        pass
    elif aug == 'rotation':
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomRotation([-int(90 * float(params[0])), int(90 * float(params[0]))]),
                # transforms.ToTensor(),
            ]
        )
    elif aug == 'crop':
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256 - int(128 * float(params[0]))),
                transforms.Resize((256, 256)),
                # transforms.ToTensor(),
            ]
        )
    elif aug == 'flip':
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(1.0),
                # transforms.ToTensor(),
            ]
        )
    elif aug == 'stretch':
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop((256 - int(128 * float(params[0])), 256)),
                transforms.Resize((256, 256)),
                # transforms.ToTensor(),
            ]
        )
    elif aug == 'blur': # remove
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.GaussianBlur(5),
                # transforms.ToTensor(),
            ]
        )
    elif aug == 'combo':
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop((256 - int(128 * float(params[0])), 256)),
                transforms.RandomHorizontalFlip(1.0),
                transforms.RandomRotation([-int(90 * float(params[0])), int(90 * float(params[0]))]),
                transforms.Resize((256, 256)),
                # transforms.ToTensor(),
            ]
        )
    elif aug == "jitter":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ColorJitter(brightness=1.0, contrast=0.5, saturation=1.0, hue=0.1),
                # transforms.ToTensor(),
            ]
        )
    # elif aug == 'edit':
    #     transform = transforms.Compose(
    #         [
    #             transforms.Resize(256),
    #             EditImage(params[0], float(params[1]), save_imgs=False),
    #             transforms.ToTensor(),
    #         ]
    #     )
    # elif aug == 'edit stretch':
    #     transform = transforms.Compose(
    #         [
    #             transforms.Resize(256),
    #             EditImage(params[0], float(params[1]), save_imgs=False),
    #             transforms.CenterCrop((256 - int(128 * float(params[2])), 256)),
    #             transforms.Resize((256, 256)),
    #             transforms.ToTensor(),
    #         ]
    #     )
    # elif aug == 'edit combo':
    #     transform = transforms.Compose(
    #         [
    #             transforms.Resize(256),
    #             EditImage("make it snow"),
    #             transforms.CenterCrop((128, 256)),
    #             transforms.RandomHorizontalFlip(1.0),
    #             transforms.RandomRotation([-90, 90]),
    #             transforms.Resize((256, 256)),
    #             transforms.ToTensor(),
    #         ]
    #     )
    # elif aug == 'gaussian':
    #     transform = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.ToTensor(),
    #         AddGaussianNoise(0., float(params[0])) # 0.05
    #         transforms.ToPILImage()
    #     ])
    elif aug == 'diffpure':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            DiffPure(steps=float(params[0])),
            transforms.ToPILImage()
        ])
    # elif aug == 'diffpure_latent':
    #     transform = transforms.Compose([
    #         transforms.Resize(256),
    #         ImageRephrase(img_size=256, strength=float(params[0]), num_passes=int(params[1]), save_imgs=save_images, fname=fname),
    #         transforms.ToTensor(),
    #     ])
    # elif aug == 'capedit':
    #     transform = transforms.Compose([
    #         transforms.Resize(256),
    #         CaptionBasedEdit(img_size=256, save_imgs=False),
    #         transforms.ToTensor(),
    #     ])
    else:
        print(f"Augmentation {aug} not implemented!!!")
        return None, None
    
    return transform, org_transform



def main():
    parser = ArgumentParser()
    parser.add_argument("--aug-type", type=str, required=True,
        help="Options: ['diffpure,{x}', 'no_aug', 'combo,{x}']")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--n-query", default=1000, type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform, _ = get_transforms(args.aug_type)

    dataset = CustomImageFolder(args.data_dir, transform=transform,
        return_id=True, shuffle=True)
    print(f"Finished loading {len(dataset)} data!")

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    with torch.no_grad():
        for i, (image, _, img_id) in enumerate(tqdm(dataset, total=args.n_query)):
            if i >= args.n_query:
                break
            image.save(os.path.join(args.out_dir, f"{img_id}.png"))
    


if __name__ == "__main__":
    main()
