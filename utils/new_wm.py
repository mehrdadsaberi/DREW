import math
import torch
from torch import nn
from torch.nn.functional import relu, sigmoid
import os


from utils.utils import WatermarkMethod




class NewWMWatermark(WatermarkMethod):
    def __init__(self, model_dir="checkpoints/wm_128_v3", img_res=256, img_channels=3, wm_bits=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = StegaStampEncoder(
            resolution=img_res,
            IMAGE_CHANNELS=img_channels,
            fingerprint_size=wm_bits
        )
        self.decoder = StegaStampDecoder(
            resolution=img_res,
            IMAGE_CHANNELS=img_channels,
            fingerprint_size=wm_bits
        )
        self.encoder.load_state_dict(torch.load(os.path.join(model_dir, 'encoder.pth')))
        self.decoder.load_state_dict(torch.load(os.path.join(model_dir, 'decoder.pth')))
        self.encoder.to(self.device)
        self.decoder.to(self.device)


    def watermark_single_image(self, image, key):
        image = image.to(self.device)
        tensor_key = torch.FloatTensor([float(bit) for bit in key]).to(self.device)
        wm_img = self.encoder(tensor_key.unsqueeze(0), image.unsqueeze(0)).squeeze(0)
        return wm_img

    def watermark_batch_images(self, images, keys):
        images = torch.stack(images).to(self.device)
        tensor_keys = torch.FloatTensor([[float(bit) for bit in string] for string in keys]).to(self.device)
        wm_imgs = self.encoder(tensor_keys, images)
        return wm_imgs
    
    def decode_single_image(self, image):
        decoded_key = self.decoder(image.unsqueeze(0)).squeeze(0)
        str_key = ""
        for x in decoded_key:
            str_key += "1" if x >= 0. else "0"
        return str_key
        


class Discriminator(nn.Module):
    def __init__(
        self,
        resolution=32,
        IMAGE_CHANNELS=1
    ):
        super(Discriminator, self).__init__()
        self.resolution = resolution
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.decoder = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, (3, 3), 2, 1),  # 16
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 8
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),  # 4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 2
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), 2, 1),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(resolution * resolution * 128 // 32 // 32, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, image):
        x = self.decoder(image)
        x = x.view(-1, self.resolution * self.resolution * 128 // 32 // 32)
        return sigmoid(self.dense(x))




class StegaStampEncoder(nn.Module):
    def __init__(
        self,
        resolution=32,
        IMAGE_CHANNELS=1,
        fingerprint_size=100,
        return_residual=False,
    ):
        super(StegaStampEncoder, self).__init__()
        self.fingerprint_size = fingerprint_size
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.return_residual = return_residual
        self.secret_dense = nn.Linear(self.fingerprint_size, 64 * 64 * IMAGE_CHANNELS)

        log_resolution = int(math.log(resolution, 2))
        assert resolution == 2 ** log_resolution, f"Image resolution must be a power of 2, got {resolution}."

        self.fingerprint_upsample = nn.Upsample(scale_factor=(2**(log_resolution-6), 2**(log_resolution-6)))
        self.conv1 = nn.Conv2d(2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)
        self.pad6 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up6 = nn.Conv2d(256, 128, 2, 1)
        self.upsample6 = nn.Upsample(scale_factor=(2, 2))
        self.conv6 = nn.Conv2d(128 + 128, 128, 3, 1, 1)
        self.pad7 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up7 = nn.Conv2d(128, 64, 2, 1)
        self.upsample7 = nn.Upsample(scale_factor=(2, 2))
        self.conv7 = nn.Conv2d(64 + 64, 64, 3, 1, 1)
        self.pad8 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up8 = nn.Conv2d(64, 32, 2, 1)
        self.upsample8 = nn.Upsample(scale_factor=(2, 2))
        self.conv8 = nn.Conv2d(32 + 32, 32, 3, 1, 1)
        self.pad9 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up9 = nn.Conv2d(32, 32, 2, 1)
        self.upsample9 = nn.Upsample(scale_factor=(2, 2))
        self.conv9 = nn.Conv2d(32 + 32 + 2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.conv10 = nn.Conv2d(32, 32, 3, 1, 1)
        self.residual = nn.Conv2d(32, IMAGE_CHANNELS, 1)

    def forward(self, fingerprint, image):
        fingerprint = relu(self.secret_dense(fingerprint))
        fingerprint = fingerprint.view((-1, self.IMAGE_CHANNELS, 64, 64))
        fingerprint_enlarged = self.fingerprint_upsample(fingerprint)
        inputs = torch.cat([fingerprint_enlarged, image], dim=1)
        conv1 = relu(self.conv1(inputs))
        conv2 = relu(self.conv2(conv1))
        conv3 = relu(self.conv3(conv2))
        conv4 = relu(self.conv4(conv3))
        conv5 = relu(self.conv5(conv4))
        up6 = relu(self.up6(self.pad6(self.upsample6(conv5))))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = relu(self.conv6(merge6))
        up7 = relu(self.up7(self.pad7(self.upsample7(conv6))))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = relu(self.conv7(merge7))
        up8 = relu(self.up8(self.pad8(self.upsample8(conv7))))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = relu(self.conv8(merge8))
        up9 = relu(self.up9(self.pad9(self.upsample9(conv8))))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = relu(self.conv9(merge9))
        conv10 = relu(self.conv10(conv9))
        residual = self.residual(conv10)
        # if not self.return_residual:
        #     residual = sigmoid(residual) * 2 - 1
        return image + residual


class StegaStampDecoder(nn.Module):
    def __init__(self, resolution=32, IMAGE_CHANNELS=1, fingerprint_size=1):
        super(StegaStampDecoder, self).__init__()
        self.resolution = resolution
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.decoder = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, (3, 3), 2, 1),  # 16
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 8
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),  # 4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 2
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), 2, 1),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(resolution * resolution * 128 // 32 // 32, 512),
            nn.ReLU(),
            nn.Linear(512, fingerprint_size),
        )

    def forward(self, image):
        x = self.decoder(image)
        x = x.view(-1, self.resolution * self.resolution * 128 // 32 // 32)
        return self.dense(x)

