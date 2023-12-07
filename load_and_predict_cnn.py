#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/ayulockin/deepimageinpainting/blob/master/Image_Inpainting_Autoencoder_Decoder_v2_0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# source: https://github.com/ayulockin/deepimageinpainting/tree/master

# # Setups, Installations and Imports
# get_ipython().system('pip install tensorflow')
# get_ipython().system('pip install wandb -q')

import tensorflow as tf
# print(tf.__version__)
from tensorflow import keras
from keras.models import load_model
# python3 -m pip install tensorflow[and-cuda]

import wandb
from wandb.keras import WandbCallback
import argparse
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import utils
import torch
# NOTE: 'module load gcc/9.2.0' is necessary for PIL to work

# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# from mpl_toolkits.axes_grid1 import ImageGrid

## Metric
def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f + y_pred_f))


# #### Data Generator with Patch Augmentation
## Ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.
class createAugment(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, dim=(64, 64), n_channels=1, mask_type="stroke"):
      'Initialization'
      self.dim = dim
      self.n_channels = n_channels
      self.mask_type = mask_type

  def data_generation(self, image):
    ## Iterate through random indexes
    # for i, idx in enumerate(idxs):
    image_copy = image.copy()

    ## Get mask associated to that image
    masked_image = self.createMask(image_copy)

    if self.n_channels == 1:
        # convert from 2D to 3D
        masked_image = tf.expand_dims(masked_image/255, axis=-1)
    else:
        masked_image = masked_image/255
    return masked_image, image/255

  def createMask(self, img):
    ## Prepare masking matrix
    if self.mask_type == "lv":
      # limited view masking - rectangle 1/3 the width of the image, centered in the middle
      mask = np.full((self.dim[0],self.dim[1],self.n_channels), 255, np.uint8)
      w = self.dim[0] // 2
      x = self.dim[0] // 2
      cv2.line(mask,(x,0),(x,self.dim[1]),(1,1,1),w)
    elif self.mask_type == "sa":
        # #   mask[:]
        # spatial aliasing masking - 1-pixel wide unmasked vertical line every m columns
        mask = np.full((self.dim[0],self.dim[1],self.n_channels), 0, np.uint8)
        thickness = np.random.randint(2, 5)
        mask[:, ::thickness, :]= 255
    elif self.mask_type == "mg":
        pattern = np.array([255,255,255,255, 0, 0, 0, 0], dtype=np.uint8)
        area = self.dim[0] * self.dim[1]
        mask = np.tile(pattern, (area) // len(pattern) + 1)[:area]
        mask = np.reshape(mask, (self.dim[0], self.dim[1], 1))
        mask = np.tile(mask, self.n_channels)
    else:
      # random "brush" strokes masking
      mask = np.full((self.dim[0],self.dim[1],self.n_channels), 255, np.uint8)
      for _ in range(np.random.randint(1, 10)):
        # Get random x locations to start line
        x1, x2 = np.random.randint(1, self.dim[0]), np.random.randint(1, self.dim[0])
        # Get random y locations to start line
        y1, y2 = np.random.randint(1, self.dim[1]), np.random.randint(1, self.dim[1])
        # Get random thickness of the line drawn
        thickness = np.random.randint(1, 3)
        # Draw black line on the white mask
        cv2.line(mask,(x1,y1),(x2,y2),(1,1,1),thickness)

    # Perform bitwise and operation to mak the image
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def predict_and_save(generator, image, model, dir, file_name, config):
    masked_image, _ = generator.data_generation(image)
    inpainted_image = model.predict(np.expand_dims(masked_image, axis=0))
    inpainted_image = inpainted_image.reshape(inpainted_image.shape[1:])


    if (config):
      np.save(os.path.join(dir, 'inpainted_orig', file_name), inpainted_image)
      np.save(os.path.join(dir, 'masked', file_name), masked_image)
      global_min = config['min']
      global_max = config['max']
      # normalization done thru (cropped_image_array - global_min) / (global_max - global_min)
      denorm_image = inpainted_image * (global_max - global_min) + global_min
      np.save(os.path.join(dir, 'inpainted_denorm', file_name), denorm_image)

    else:
      image = torch.from_numpy(np.transpose(image, (2,0,1)))
      masked_image = torch.from_numpy(1-np.transpose(masked_image, (2,0,1)))
      inpainted_image = torch.from_numpy(1-np.transpose(inpainted_image, (2,0,1)))
      utils.display_results(None, image, masked_image, inpainted_image, os.path.join(dir,  os.path.splitext(file_name)[0] + '.png'))


# # Testing on images

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
if __name__ == '__main__':
    print('[INFO]', tf.config.experimental.list_physical_devices('GPU'))
    parser = argparse.ArgumentParser(description="cnn_predictor")
    parser.add_argument('--image-size', type=int, default=32, help='Image height and width (default: 32)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str,help='Path to parent directory data (ex: /groups/mlprojects/pat/pat_norm_crop)')
    parser.add_argument('--model-path', type=str, default='wandb/latest-run/files/model-best.h5', help='Path to model to load (default: best model of latest run)')
    parser.add_argument('--result_directory', type=str, help='Name of directory to save the results to.')

    group = parser.add_mutually_exclusive_group()

    group.add_argument('--limited-view', action='store_true', help='This will use a square mask instead of random stroke masks')
    group.add_argument('--spatial-aliasing', action='store_true', help='This will mimic spatial aliasing instead of random stroke masks')
    group.add_argument('--maskgit-spatial-aliasing', action='store_true', help='This will mimic spatial aliasing used in maskgit (4 pixel wide columns)')

    args = parser.parse_args()

    print(tf.__version__)

    x_test = [] # image arrays
    file_names = [] # file names
    print("Loading data...")
    for f in tqdm(os.listdir(os.path.join(args.dataset_path, 'test'))):
        if f.endswith('.npy'):
            x_test.append(np.load(os.path.join(args.dataset_path, 'test', f)) * 255)
            file_names.append(f)
        elif f.endswith('.jpg'):
            img = Image.open(os.path.join(args.dataset_path, 'test', f)).resize((256, 256))
            img_array = np.asarray(img) * 255
            x_test.append(img_array)
            file_names.append(f)

    x_test = np.asarray(x_test, dtype=object).astype(np.uint8)

    # Load config -- for denormalization
    config_path = os.path.join(args.dataset_path, 'config.json')
    config = None
    if (os.path.isfile(config_path)):
      json_file = open(config_path)
      config = json.load(json_file)
      json_file.close()

    mask_type = "stroke"
    if args.limited_view:
      mask_type = "lv"
    elif args.spatial_aliasing:
      mask_type = "sa"
    elif args.maskgit_spatial_aliasing:
      mask_type = "mg"

    ## Prepare training and testing mask-image pair generator
    testgen = createAugment(dim=(args.image_size, args.image_size), n_channels=args.image_channels, mask_type=mask_type)

    model = load_model(args.model_path, custom_objects={'dice_coef': dice_coef})
    # print number of parameters
    print(model.summary())
    print(model.count_params())

    os.makedirs(os.path.join(args.result_directory, 'inpainted_orig'), exist_ok=True)
    os.makedirs(os.path.join(args.result_directory, 'inpainted_denorm'), exist_ok=True)
    os.makedirs(os.path.join(args.result_directory, 'masked'), exist_ok=True)


    print("Inpainting and saving...")
    for i, image in enumerate(tqdm(x_test, total=x_test.shape[0])):
       predict_and_save(testgen, image, model, args.result_directory, file_names[i], config)
