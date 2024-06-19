from trustmark.trustmark import TrustMark
from PIL import Image
import os
import sys

from utils.utils import WatermarkMethod


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Trustmark(WatermarkMethod):
    def __init__(self, wm_bits):
        with HiddenPrints():
            self.tm = TrustMark(model_type='Q', use_ECC=False, secret_len=wm_bits, verbose=False)

    def watermark_single_image(self, image, key): # PIL Image input
        with HiddenPrints():
            wm_img = self.tm.encode(image, key, MODE="binary")
        return wm_img

    def watermark_batch_images(self, images, keys):
        with HiddenPrints():
            return [self.watermark_single_image(image, key) for image, key in zip(images, keys)]
    
    def decode_single_image(self, image):
        with HiddenPrints():
            str_key, _, _ = self.tm.decode(image, MODE="binary")
        return str_key[0].astype(int)
        
