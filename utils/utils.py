from PIL import Image
from torch.utils.data import Dataset
import glob
import os
import random
import torch
import pickle
from abc import ABC, abstractmethod
import argparse
import yaml
import numpy as np
import math
from tqdm import tqdm

from python_polar_coding.polar_codes import FastSSCPolarCodec

from DiffPure.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


def calc_match_rate(x: str, y: str):
    assert len(x) == len(y)
    return sum([x[i] == y[i] for i in range(len(x))]) / len(x)

class WatermarkMethod(ABC):
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def watermark_single_image(self, image, key):
        r""" Recieves PIL image and string binary key, Returns watermarked PIL image."""
        pass


class WatermarkDataset:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def create_clusters(self, n_bits, n_clusters, mode="random"):
        assert n_clusters <= len(self.dataset)

        self.n_bits = n_bits
        self.n_clusters = n_clusters

        # Distribute dataset items evenly across clusters, adjusting for remainder.
        cluster_sizes = [len(self.dataset) // self.n_clusters for _ in range(self.n_clusters)]
        for i in range(len(self.dataset) % self.n_clusters):
            cluster_sizes[i] += 1

        # Prepare data structures for binary codes and cluster assignments.
        self.img_id_to_cluster_id = {}  # Map image IDs to cluster binary codes
        self.cluster_id_to_img_id = {}  # Map cluster binary codes to image IDs

        # Generate binary codes and assign data items to clusters.
        self.cluster_id_to_code = self.get_codes(self.n_bits, self.n_clusters, mode=mode)
        random.seed(0)  
        data_ids = list(range(len(self.dataset)))
        random.shuffle(data_ids)  # Randomize data assignment to clusters.

        # Assign data items to clusters
        cur_start = 0
        for i, (cl_id, code) in enumerate(self.cluster_id_to_code.items()):
            self.cluster_id_to_img_id[cl_id] = []
            for j in data_ids[cur_start: cur_start + cluster_sizes[i]]:
                img_id = self.dataset.img_ids[j]
                self.img_id_to_cluster_id[img_id] = cl_id
                self.cluster_id_to_img_id[cl_id].append(img_id)
            cur_start += cluster_sizes[i]
        
        assert cur_start == len(self.dataset)  # Verify all items are assigned.

    def store_cluster_info(self, file_dir):
        data = (self.n_bits, self.n_clusters, self.img_id_to_cluster_id, self.cluster_id_to_img_id, self.cluster_id_to_code)
        with open(file_dir, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_cluster_info(self, file_dir):
        with open(file_dir, 'rb') as handle:
            data = pickle.load(handle)
        self.n_bits, self.n_clusters, self.img_id_to_cluster_id, self.cluster_id_to_img_id, self.cluster_id_to_code = data

    def get_codes(self, n_bits, n_clusters, mode="random", seed=0):
        """Generate n_clusters binary codes with n_bits bits."""
        random.seed(0)
        codes = {}
        # Assign random binary codes to the clusters
        # if mode == "random": #TODO WRONG
        #     for i in range(n_clusters):
        #         code = ""
        #         for j in range(n_bits):
        #             code += str(random.randint(0, 1))
        #         codes.append(code)
        if mode == "ecc_polar":
            assert (n_clusters & (n_clusters-1) == 0) and n_clusters != 0 # check if n_cluster is power of 2
            assert (n_bits & (n_bits-1) == 0) and n_bits != 0 # check if n_bits is power of 2

            N = n_bits
            K = int(math.log2(n_clusters))

            codec = FastSSCPolarCodec(N=N, K=K, design_snr=0.0)

            for i in tqdm(range(n_clusters)):
                # convert i to binary message with length log(n_clusters)
                org_key = [int(x) for x in bin(i)[2:]]
                while (len(org_key) < K):
                    org_key = [0] + org_key
                org_key = np.array(org_key)

                encoded_key = codec.encode(org_key)
                
                code = ""
                for x in encoded_key:
                    code += str(int(x))
                codes["".join([str(x) for x in org_key])] = code
        
        elif mode == "ecc_polar_100":
            assert (n_clusters & (n_clusters-1) == 0) and n_clusters != 0 # check if n_cluster is power of 2
            assert n_bits == 100

            N = n_bits
            K = int(math.log2(n_clusters))

            K_64 = math.ceil(K * 2 / 3)
            K_32 = K - K_64

            codec_64 = FastSSCPolarCodec(N=64, K=K_64, design_snr=0.0)
            codec_32 = FastSSCPolarCodec(N=32, K=K_32, design_snr=0.0)

            for i in tqdm(range(n_clusters)):
                # convert i to binary message with length log(n_clusters)
                org_key = [int(x) for x in bin(i)[2:]]
                while (len(org_key) < K):
                    org_key = [0] + org_key
                org_key = np.array(org_key)


                encoded_key_64 = codec_64.encode(org_key[:K_64])
                encoded_key_32 = codec_32.encode(org_key[K_64:])
                
                code = ""
                for x in encoded_key_64:
                    code += str(int(x))
                for x in encoded_key_32:
                    code += str(int(x))
                assert len(code) == 96
                code += "0000"
                codes["".join([str(x) for x in org_key])] = code

        else:
            raise Exception(f"Code generation mode {mode} is not implemented!")
        return codes


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None, data_cnt=-1, y=0, shuffle=False,
        return_fname=False, return_id=False, fname_format=None):
        self.data_dir = data_dir
        if not fname_format:
            fname_format = "*"
            self.filenames = glob.glob(os.path.join(data_dir, f"*/{fname_format}.png"))
            self.filenames.extend(glob.glob(os.path.join(data_dir, f"*/{fname_format}.jpeg")))
            self.filenames.extend(glob.glob(os.path.join(data_dir, f"*/{fname_format}.JPEG")))
            self.filenames.extend(glob.glob(os.path.join(data_dir, f"*/{fname_format}.jpg")))
            self.filenames.extend(glob.glob(os.path.join(data_dir, f"{fname_format}.png")))
            self.filenames.extend(glob.glob(os.path.join(data_dir, f"{fname_format}.jpeg")))
            self.filenames.extend(glob.glob(os.path.join(data_dir, f"{fname_format}.JPEG")))
            self.filenames.extend(glob.glob(os.path.join(data_dir, f"{fname_format}.jpg")))
        else:
            self.filenames = glob.glob(os.path.join(data_dir, fname_format))
        
        if shuffle:
            random.seed(17)
            random.shuffle(self.filenames)
        else:
            self.filenames.sort()

        if data_cnt != -1:
            self.filenames = self.filenames[:data_cnt]
        if data_dir[-1] != '/':
            data_dir += '/'
        self.img_ids = [x.replace(data_dir, '').replace('/', '_').split('.')[0] for x in self.filenames]
        self.transform = transform
        self.y = y
        self.return_fname = return_fname
        self.return_id = return_id

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_id = self.img_ids[idx]
        image = Image.open(filename).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.return_fname:
            return image, self.y, filename
        elif self.return_id:
            return image, self.y, img_id
        return image, self.y

    def __len__(self):
        return len(self.filenames)

class CustomDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if shuffle:
            random.seed(17)
            random.shuffle(self.indices)

    def __iter__(self):
        self.cur_idx = 0
        return self

    def __next__(self):
        if self.cur_idx >= len(self.dataset):
            raise StopIteration
        indices = self.indices[self.cur_idx:self.cur_idx + self.batch_size]
        batch = [self.dataset[idx] for idx in indices]
        self.cur_idx += self.batch_size
        
        zipped_batch = list(zip(*batch))
        batch_lists = [list(items) for items in zipped_batch]
        
        return batch_lists

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    

class DiffPure():
    def __init__(self, steps=0.4, save_imgs=False, fname="base"):
        with open('DiffPure/configs/imagenet.yml', 'r') as f:
            config = yaml.safe_load(f)
        self.config = dict2namespace(config)
        self.runner = GuidedDiffusion(self.config, t = int(steps * int(self.config.model.timestep_respacing)), model_dir = 'DiffPure/pretrained/guided_diffusion')
        self.steps = steps
        self.save_imgs = save_imgs
        self.cnt = 0
        self.fname = fname

        if self.save_imgs:
            save_dir = f'./diffpure_images/{self.fname}/{self.steps}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    def __call__(self, img):
        img_pured, img_noisy = self.runner.image_editing_sample((img.unsqueeze(0) - 0.5) * 2)
        img_noisy = (img_noisy.squeeze(0).to(img.dtype).to("cpu") + 1) / 2
        img_pured = (img_pured.squeeze(0).to(img.dtype).to("cpu") + 1) / 2
        if self.save_imgs:
            save_dir = f'./diffpure_images/{self.fname}/{self.steps}'
            save_image(img, os.path.join(save_dir, f'{self.cnt}.png'))
            save_image(img_noisy, os.path.join(save_dir, f'{self.cnt}_noisy.png'))
            save_image(img_pured, os.path.join(save_dir, f'{self.cnt}_pured.png'))
            self.cnt += 1
        return img_pured
    
    def __repr__(self):
        return self.__class__.__name__ + '(steps={})'.format(self.steps)



def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class GuidedDiffusion(torch.nn.Module):
    def __init__(self, config, t, device=None, model_dir='pretrained/guided_diffusion'):
        super().__init__()
        # self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.sample_step = 1
        self.t = t

        # load model
        model_config = model_and_diffusion_defaults()
        model_config.update(vars(self.config.model))
        # print(f'model_config: {model_config}')
        model, diffusion = create_model_and_diffusion(**model_config)
        model.load_state_dict(torch.load(f'{model_dir}/256x256_diffusion_uncond.pt', map_location='cpu'))
        model.requires_grad_(False).eval().to(self.device)

        if model_config['use_fp16']:
            model.convert_to_fp16()

        self.model = model
        self.diffusion = diffusion
        self.betas = torch.from_numpy(diffusion.betas).float().to(self.device)

    def image_editing_sample(self, img, bs_id=0, tag=None):
        with torch.no_grad():
            assert isinstance(img, torch.Tensor)
            batch_size = img.shape[0]

            # if tag is None:
            #     tag = 'rnd' + str(random.randint(0, 10000))
            # out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

            assert img.ndim == 4, img.ndim
            img = img.to(self.device)
            x0 = img

            # if bs_id < 2:
            #     os.makedirs(out_dir, exist_ok=True)
            #     tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'original_input.png'))

            xs = []
            xts = []
            for it in range(self.sample_step):
                e = torch.randn_like(x0)
                total_noise_levels = self.t
                a = (1 - self.betas).cumprod(dim=0)
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                xts.append(x.clone())

                # if bs_id < 2:
                #     tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'init_{it}.png'))

                for i in reversed(range(total_noise_levels)):
                    t = torch.tensor([i] * batch_size, device=self.device)

                    x = self.diffusion.p_sample(self.model, x, t,
                                                clip_denoised=True,
                                                denoised_fn=None,
                                                cond_fn=None,
                                                model_kwargs=None)["sample"]

                    # added intermediate step vis
                    # if (i - 99) % 100 == 0 and bs_id < 2:
                    #     tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'noise_t_{i}_{it}.png'))

                x0 = x

                # if bs_id < 2:
                #     torch.save(x0, os.path.join(out_dir, f'samples_{it}.pth'))
                #     tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'samples_{it}.png'))

                xs.append(x0)

            return torch.cat(xs, dim=0), torch.cat(xts, dim=0)






