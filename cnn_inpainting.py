#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/ayulockin/deepimageinpainting/blob/master/Image_Inpainting_Autoencoder_Decoder_v2_0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Setups, Installations and Imports
# get_ipython().system('pip install tensorflow')
# get_ipython().system('pip install wandb -q')

import tensorflow as tf
# print(tf.__version__)
from tensorflow import keras
# python3 -m pip install tensorflow[and-cuda]

import wandb
from wandb.keras import WandbCallback
import argparse
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
# NOTE: 'module load gcc/9.2.0' is necessary for PIL to work

# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# from mpl_toolkits.axes_grid1 import ImageGrid

# # Prepare Dataset
def load_and_crop(data_dir, file_name, original_width, original_height, target_size):
    '''
    This function was used to create the dataset in '/groups/mlprojects/pat/pat_cnn'.
    It is not currently used anywhere below.
    '''
    print(f"Processing {file_name}")
    # Load the image from the .npy file
    image = np.load(os.path.join(data_dir, file_name))

    # Randomly crop the image to the target size
    left = np.random.randint(0, original_width - target_size[0] + 1)
    top = np.random.randint(0, original_height - target_size[1] + 1)
    right = left + target_size[0]
    bottom = top + target_size[1]
    # cropped_image = pil_image.crop((left, top, right, bottom))
    cropped_image = image[top:bottom, left:right]

    # Reshape to be expected size
    cropped_image_array = np.array(cropped_image).reshape(*target_size, 1)
    # normalize to be from 0 to 255 since that is what is expected below
    normalized_image = (cropped_image_array - cropped_image_array.min()) * (255.0 / (cropped_image_array.max() - cropped_image_array.min()))

    # Save the cropped image array to a new .npy file
    new_file_name = f"cropped_{file_name}"
    new_dir = '/groups/mlprojects/pat/pat_new'
    np.save(os.path.join(new_dir, new_file_name), cropped_image_array)
    # image needs to be unit8 and not float32 in order to do masking properly
    return normalized_image.astype('uint8')



# #### Data Generator with Patch Augmentation
## Ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.
class createAugment(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, X, y, batch_size=32, dim=(64, 64), n_channels=1, shuffle=True, mask_type="stroke"):
      'Initialization'
      self.batch_size = batch_size
      self.X = X
      self.y = y
      self.dim = dim
      self.n_channels = n_channels
      self.shuffle = shuffle
      self.mask_type = mask_type

      self.on_epoch_end()

  def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.X) / self.batch_size))

  def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

      # Generate data
      return self.__data_generation(indexes)

  def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.X))
      if self.shuffle:
          np.random.shuffle(self.indexes)

  def __data_generation(self, idxs):
    # X_batch is a matrix of masked images used as input
    X_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Masked image
    # y_batch is a matrix of original images used for computing error from reconstructed image
    y_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # Original image

    ## Iterate through random indexes
    for i, idx in enumerate(idxs):
      image_copy = self.X[idx].copy()

      ## Get mask associated to that image
      masked_image = self.__createMask(image_copy)

      if self.n_channels == 1:
        # convert from 2D to 3D
        X_batch[i,] = tf.expand_dims(masked_image/255, axis=-1)
      else:
        X_batch[i,] = masked_image/255
      y_batch[i] = self.y[idx]/255

    return X_batch, y_batch

  def __createMask(self, img):
    ## Prepare masking matrix
    mask = np.full((self.dim[0],self.dim[1],self.n_channels), 255, np.uint8)
    if self.mask_type == "lv":
      # limited view masking - rectangle 1/3 the width of the image, centered in the middle
      w = self.dim[0] // 3
      x = self.dim[0] // 2
      cv2.line(mask,(x,0),(x,self.dim[1]),(1,1,1),w)
    elif self.mask_type == "sa":
        # spatial aliasing masking - 1-pixel wide vertical line every n columns
        thickness = 1
        multiplier = np.random.randint(3, 6)
        x_coords = np.linspace(0, self.dim[0] - 1, self.dim[0] // (thickness * multiplier), dtype=np.uint8)
        for x in x_coords:
        # Draw black line on the white mask
          cv2.line(mask,(x,0),(x,self.dim[1]),(1,1,1),thickness)
    else:
      # random "brush" strokes masking
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

## For more information into formulation: https://www.youtube.com/watch?v=AZr64OxshLo
## Metric
def dice_coef(y_true, y_pred):
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.backend.sum(y_true_f + y_pred_f))

class inpaintingModel:
  '''
  Build UNET like model for image inpainting task.
  '''
  def prepare_model(self, input_size=(64,64,1)):
    inputs = keras.layers.Input(input_size)

    conv1, pool1 = self.__ConvBlock(32, (3,3), (2,2), 'relu', 'same', inputs)
    conv2, pool2 = self.__ConvBlock(64, (3,3), (2,2), 'relu', 'same', pool1)
    conv3, pool3 = self.__ConvBlock(128, (3,3), (2,2), 'relu', 'same', pool2)
    conv4, pool4 = self.__ConvBlock(256, (3,3), (2,2), 'relu', 'same', pool3)

    conv5, up6 = self.__UpConvBlock(512, 256, (3,3), (2,2), (2,2), 'relu', 'same', pool4, conv4)
    conv6, up7 = self.__UpConvBlock(256, 128, (3,3), (2,2), (2,2), 'relu', 'same', up6, conv3)
    conv7, up8 = self.__UpConvBlock(128, 64, (3,3), (2,2), (2,2), 'relu', 'same', up7, conv2)
    conv8, up9 = self.__UpConvBlock(64, 32, (3,3), (2,2), (2,2), 'relu', 'same', up8, conv1)

    conv9 = self.__ConvBlock(32, (3,3), (2,2), 'relu', 'same', up9, False)

    outputs = keras.layers.Conv2D(input_size[2], (3, 3), activation='sigmoid', padding='same')(conv9)

    return keras.models.Model(inputs=[inputs], outputs=[outputs])

  def __ConvBlock(self, filters, kernel_size, pool_size, activation, padding, connecting_layer, pool_layer=True):
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
    if pool_layer:
      pool = keras.layers.MaxPooling2D(pool_size)(conv)
      return conv, pool
    else:
      return conv

  def __UpConvBlock(self, filters, up_filters, kernel_size, up_kernel, up_stride, activation, padding, connecting_layer, shared_layer):
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
    up = keras.layers.Conv2DTranspose(filters=up_filters, kernel_size=up_kernel, strides=up_stride, padding=padding)(conv)
    up = keras.layers.concatenate([up, shared_layer], axis=3)

    return conv, up

# # Train

class PredictionLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super(PredictionLogger, self).__init__()

    def on_epoch_end(self, logs, epoch):
        sample_idx = 54
        sample_images, sample_labels = testgen[sample_idx]

        images = []
        labels = []
        predictions = []

        for i in range(32):
            inpainted_image = self.model.predict(np.expand_dims(sample_images[i], axis=0))

            images.append(sample_images[i])
            labels.append(sample_labels[i])
            predictions.append(inpainted_image.reshape(inpainted_image.shape[1:]))

        wandb.log({"images": [wandb.Image(image)
                              for image in images]})
        wandb.log({"labels": [wandb.Image(label)
                              for label in labels]})
        wandb.log({"predictions": [wandb.Image(inpainted_image)
                              for inpainted_image in predictions]})



# # Testing on images

# fig, axs = plt.subplots(nrows=rows, ncols=3, figsize=(6, 2*rows))

# for i in range(32):
#   impainted_image = model.predict(sample_images[i].reshape((1,)+sample_images[i].shape))
#   axs[i][0].imshow(sample_labels[i])
#   axs[i][1].imshow(sample_images[i])
#   axs[i][2].imshow(impainted_image.reshape(impainted_image.shape[1:]))

# plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
# import subprocess
if __name__ == '__main__':
    print('[INFO]', tf.config.experimental.list_physical_devices('GPU'))
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--image-size', type=int, default=32, help='Image height and width (default: 32)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='cifar', help='Path to data (default: uses cifar dataset)')
    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 50)')
    parser.add_argument('--experiment-name', type=str, help='Name of experiment. This will be used to create a folder in results/ and checkpoints/')

    group = parser.add_mutually_exclusive_group()

    group.add_argument('--limited_view', action='store_true', help='This will use a square mask instead of random stroke masks')
    group.add_argument('--spatial_aliasing', action='store_true', help='This will mimic spatial aliasing instead of random stroke masks')


    args = parser.parse_args()

    print(tf.__version__)
    # set your wandb API key (https://docs.wandb.ai/quickstart)
    # in ~/.bashrc, put
    # export WANDB_API_KEY="your_api_key_here"
    wandb.login(key=os.environ["WANDB_API_KEY"])

    if args.dataset_path != 'cifar':
        # data_dir = "/groups/mlprojects/pat/pat_cnn"
        # data = [load_and_crop(data_dir, f, original_width, original_height, target_size) for f in os.listdir(data_dir) if f.endswith('.npy')]
        data = [np.load(os.path.join(args.dataset_path, f)) for f in os.listdir(args.dataset_path) if f.endswith('.npy')]

        x_train, x_test = train_test_split(data, train_size=0.8, random_state=42)
        x_train = np.array(x_train)
        x_test = np.array(x_test)
    else:
       (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
       args.image_channels = 3
       args.image_size = 32

    # this is just to test that PIL imported properly -- otherwise you won't find out until the end of an epoch
    Image.fromarray(np.squeeze(x_train[0]))

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    mask_type = "stroke"
    if args.limited_view:
      mask_type = "lv"
    elif args.spatial_aliasing:
      mask_type = "sa"

    ## Prepare training and testing mask-image pair generator
    traingen = createAugment(x_train, x_train, dim=(args.image_size, args.image_size), n_channels=args.image_channels, mask_type=mask_type)
    testgen = createAugment(x_test, x_test, shuffle=False, dim=(args.image_size, args.image_size), n_channels=args.image_channels, mask_type=mask_type)

    ## Save examples of what the masking looks like
    sample_idx = 99 ## Change this to see different batches
    sample_masks, sample_labels = traingen[sample_idx]
    sample_images = [None]*(len(sample_masks)+len(sample_labels))
    sample_images[::2] = sample_labels
    sample_images[1::2] = sample_masks
    fig = plt.figure(figsize=(16., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(4, 8),  # creates 2x2 grid of axes
                    axes_pad=0.3,  # pad between axes in inch.
                    )
    for ax, image in zip(grid, sample_images):
      ax.imshow(image)
    os.makedirs("cnn", exist_ok=True)
    fig.savefig(f'cnn/cnn_masks_{args.experiment_name}.png')

    keras.backend.clear_session()
    model = inpaintingModel().prepare_model(input_size=(args.image_size, args.image_size, args.image_channels))
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[dice_coef])
    # keras.utils.plot_model(model, show_shapes=True, dpi=76, to_file='model_v1.png')

    # lmk if you want to be added to a shared project: https://wandb.ai/mshao0/pat-inpainting/overview
    wandb.init(project="pat-inpainting", name=args.experiment_name, config=args)
    _ = model.fit(traingen,
            validation_data=testgen,
            epochs=args.epochs,
            steps_per_epoch=len(traingen),
            validation_steps=len(testgen),
            # use_multiprocessing=True,
            callbacks=[WandbCallback(),
                        PredictionLogger()])

    ## Examples
    # rows = 32
    # sample_idx = 54
    # sample_images, sample_labels = traingen[sample_idx]
