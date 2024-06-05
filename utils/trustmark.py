from trustmark.trustmark import TrustMark
from PIL import Image

from utils.utils import WatermarkMethod


class Trustmark(WatermarkMethod):
    def __init__(self, wm_bits):
        self.tm = TrustMark(model_type='Q', use_ECC=False, secret_len=wm_bits)

    def watermark_single_image(self, image, key): # PIL Image input
        wm_img = self.tm.encode(image, key, MODE="binary")
        return wm_img

    def watermark_batch_images(self, images, keys):
        return [self.watermark_single_image(image, key) for image, key in zip(images, keys)]
    
    def decode_single_image(self, image):
        str_key, _, _ = self.tm.decode(image, MODE="binary")
        return str_key[0].astype(int)
        
