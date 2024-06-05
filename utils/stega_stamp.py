# import bchlib
import glob
import os
from PIL import Image,ImageOps
import numpy as np
import tensorflow.compat.v1 as tf
# import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import random
from tqdm import tqdm
from torchvision import transforms
import cv2
import torch
from time import time

from utils.utils import WatermarkMethod




class StegaStampWatermark(WatermarkMethod):
    def __init__(self, model_dir="checkpoints/stegaStamp/stegastamp_pretrained"):
        self.model_dir = model_dir
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Enable dynamic GPU memory allocation
        self.sess = tf.InteractiveSession(graph=tf.Graph(), config=config)


        self.model = tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], self.model_dir)

        input_secret_name = self.model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
        input_image_name = self.model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
        self.input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
        self.input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

        output_stegastamp_name = self.model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
        self.output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)

        output_secret_name = self.model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
        self.output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

        print("Available Devices:", tf.config.list_physical_devices())


    def watermark_single_image(self, image, key):
        image = np.array(ImageOps.fit(image, (400, 400)),dtype=np.float32)
        image /= 255.

        list_key = [int(x) for x in key]

        feed_dict = {self.input_secret:[list_key],
                        self.input_image:[image]}

        wm_image = self.sess.run([self.output_stegastamp], feed_dict=feed_dict)
        rescaled = (wm_image[0][0] * 255).astype(np.uint8)
        im = Image.fromarray(np.array(rescaled))

        return im

    def watermark_batch_images(self, images, keys):
        # Assuming images is a list of PIL Image objects and keys is a list of binary strings
        processed_images = [np.array(ImageOps.fit(image, (400, 400)), dtype=np.float32) / 255. for image in images]
        list_keys = [[int(x) for x in key] for key in keys]

        feed_dict = {self.input_secret: list_keys, self.input_image: processed_images}

        wm_images = self.sess.run([self.output_stegastamp], feed_dict=feed_dict)
        
        rescaled_images = [(img * 255).astype(np.uint8) for img in wm_images[0]]
        pil_images = [Image.fromarray(img) for img in rescaled_images]

        return pil_images
    
    def decode_single_image(self, image):
        image = np.array(ImageOps.fit(image, (400, 400)),dtype=np.float32)
        image /= 255.
        feed_dict = {self.input_image:[image]}
        secret = self.sess.run([self.output_secret],feed_dict=feed_dict)[0][0]
        secret = "".join([str(int(x)) for x in secret])
        return secret
        

def run_stega_stamp(dataset=None, dataset_name='imagenet', out_dir='images/imagenet/stegaStamp',
    secret=None):
    # secret must be a str of 100 bits of 0s and 1s
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--save_dir', type=str)

    args = parser.parse_args(['--model', 'checkpoints/stegaStamp/stegastamp_pretrained',
                              '--save_dir', out_dir])

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

    # bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    # data = bytearray(args.secret + ' '*(12-len(args.secret)), 'utf-8')
    # ecc = bch.encode(data)
    # packet = data + ecc

    # packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in secret]
    # secret.extend([0,0,0,0])

    print(len(secret))

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        for i, (img_tensor, label) in tqdm(enumerate(dataset)):
            image = img_tensor.numpy().transpose(1, 2, 0)
            image = cv2.resize(image, (400, 400), interpolation = cv2.INTER_LINEAR)

            feed_dict = {input_secret:[secret],
                         input_image:[image]}

            hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)

            rescaled = (hidden_img[0] * 255).astype(np.uint8)


            im = Image.fromarray(np.array(rescaled))
            im.save(os.path.join(args.save_dir, f'{dataset.img_ids[i]}.png'))
