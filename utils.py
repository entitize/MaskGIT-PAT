import os
import random
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        for image in self.images:
            if image.endswith("Store"):
                self.images.remove(image)
                break
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        if image_path.endswith("npy"):
            image = np.load(image_path)
            assert image.shape[1] - self.size >= 0 and image.shape[0] - self.size >= 0, f"Image Size {self.size} is too large for images {image.shape}"
            x = random.randint(0, image.shape[1] - self.size)
            y = random.randint(0, image.shape[0] - self.size)
            image = image[y:y+self.size, x:x+self.size]
            maxValue = np.max(image)
            minValue = np.min(image)
            image = (image - minValue) * 2 / (maxValue - minValue) - 1.0
            image = np.expand_dims(image.astype(np.float32), axis=2)
        else:
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = np.array(image).astype(np.uint8)
            image = self.preprocessor(image=image)["image"]
            image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_data(args):
    train_data = ImagePaths(args.dataset_path, size=args.image_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    return train_loader


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images: dict):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    new_sample = images["new_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(new_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()
