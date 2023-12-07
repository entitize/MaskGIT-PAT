import os
import random
import albumentations
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import utils as vutils
from torchvision import transforms



# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #
def add_center_highlight(image, mask_array, border_size=1, border_color=1.0):
    if (mask_array is None):
        return image
        
    # Create a copy of the image to draw the border
    bordered_image = image.clone()

    # Draw border
    for row in mask_array:
        x_min, x_max, y_min, y_max = row
        bordered_image[:, y_max-border_size:y_max, x_min:x_max] = border_color
        bordered_image[:, y_min:y_min+border_size, x_min:x_max] = border_color
        bordered_image[:, y_min:y_max, x_min:x_min+border_size] = border_color
        bordered_image[:, y_min:y_max, x_max-border_size:x_max] = border_color
        
    return bordered_image

def normalize_image(image):
    # Normalize from [-1, 1] to [0, 1]
    if (image.min() >= 0):
        return image
    return (image + 1) / 2


def display_results(mask_array, original, masked, inpainted, result_path):
    sample_image_squeezed = original.squeeze(0) if original.dim() == 4 else original
    masked_image_squeezed = masked.squeeze(0) if masked.dim() == 4 else masked
    inpainted_image_squeezed = inpainted.squeeze(0) if inpainted.dim() == 4 else inpainted

    sample_image_highlighted = normalize_image(sample_image_squeezed)
    masked_image_highlighted = normalize_image(add_center_highlight(masked_image_squeezed, mask_array))
    inpainted_image_highlighted = normalize_image(inpainted_image_squeezed)

    images = [sample_image_highlighted, masked_image_highlighted, inpainted_image_highlighted]
    padding = 10
    grid = vutils.make_grid(images, nrow=3, padding=padding)
    grid_pil = transforms.ToPILImage()(grid)
    draw = ImageDraw.Draw(grid_pil)
    font = ImageFont.load_default()
    titles = ["Original", "Masked", "Inpainted"]
    for i, title in enumerate(titles):
        x = i * (original.shape[-1] + padding) + padding
        y = 0
        draw.text((x, y), title, (255, 255, 255), font=font) 
    grid_pil.save(result_path)


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
            image = np.squeeze(image)
            assert image.shape[1] - self.size >= 0 and image.shape[0] - self.size >= 0, f"Image Size {self.size} is too large for images {image.shape}"
            # random crop
            if (image.shape[1] - self.size > 0 or image.shape[0] - self.size > 0):
                x = random.randint(0, image.shape[1] - self.size)
                y = random.randint(0, image.shape[0] - self.size)
                image = image[y:y+self.size, x:x+self.size]
            # normalize
            maxValue = np.max(image)
            minValue = np.min(image)
            image = (image - minValue) * 2 / (maxValue - minValue) - 1.0
            image = np.expand_dims(image.astype(np.float32), axis=2)
            image = image.astype(np.float32)
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
