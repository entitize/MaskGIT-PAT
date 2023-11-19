from transformer import VQGANTransformer
import torch
import numpy as np
import os
import argparse
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from utils import load_data, plot_images
from torchvision import utils as vutils
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import wandb
import json

class Painting:

    def __init__(self, args):
        self.args = args
        self.model = VQGANTransformer(args).to(device=args.device)
        self.model.load_state_dict(torch.load(args.transformer_checkpoint_path))
    
    def normalize_image(self, x):
        return 

    def plot_index_map(self, idxs_map, full_path, filename):
        # Create a new figure
        fig = plt.figure(figsize=(10, 10))

        # Calculate the number of rows and columns for the subplot grid
        num_images = len(idxs_map)
        num_cols = round(math.sqrt(num_images))
        num_rows = num_images // num_cols
        if num_images % num_cols: num_rows += 1

        for i, (idx_name, idx_grid) in enumerate(idxs_map.items()):
            # Normalize the indices and convert to RGB
            normalized_indices = idx_grid / (self.args.num_codebook_vectors + 2)
            image = plt.get_cmap('turbo')(normalized_indices)[:, :, :3]
            p = int(math.sqrt(self.args.num_image_tokens))
            image = image.reshape(p, p, 3)

            # Add a subplot for this image
            ax = fig.add_subplot(num_rows, num_cols, i+1)
            ax.imshow(image)
            ax.set_title(idx_name)
            ax.axis('off')

        # Save the figure
        plt.savefig(os.path.join(full_path, f'{filename}_index_map.jpg'))
        plt.close(fig)
    
    def run_inpainting(self, dataset, mask_array, filename):

        for i in tqdm(range(self.args.num_inpainting_images)):
            sample_image = next(dataset).to(device=self.args.device)

            # custom inpainting (not using provided inpainting function), no blending
            idxs_map, masked_image, inpainted_image = self.model.custom_inpainting(sample_image, mask_array)

            # create a directory for each image
            os.makedirs(os.path.join(self.args.inpainting_results_dir, f"image_{i}"), exist_ok=True)
            full_path = os.path.join(self.args.inpainting_results_dir, f"image_{i}")

            self.plot_index_map(idxs_map, full_path, filename)
            
            sample_image_squeezed = sample_image.squeeze(0)
            masked_image_squeezed = masked_image.squeeze(0)
            inpainted_image_squeezed = inpainted_image.squeeze(0)

            def add_center_highlight(image, border_size=1, border_color=1.0):
                # Assuming image is a single-channel grayscale image with shape [1, Height, Width]

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
                return (image + 1) / 2

            np.save(os.path.join(full_path, f"{filename}_original.npy"), sample_image_squeezed.clone().cpu().numpy())
            np.save(os.path.join(full_path, f"{filename}_masked.npy"), masked_image_squeezed.clone().cpu().numpy())
            np.save(os.path.join(full_path, f"{filename}_inpainted.npy"), inpainted_image_squeezed.clone().cpu().numpy())

            sample_image_highlighted = normalize_image(add_center_highlight(sample_image_squeezed))
            masked_image_highlighted = normalize_image(add_center_highlight(masked_image_squeezed))
            inpainted_image_highlighted = normalize_image(add_center_highlight(inpainted_image_squeezed))

            images = [sample_image_highlighted, masked_image_highlighted, inpainted_image_highlighted]
            padding = 10
            grid = vutils.make_grid(images, nrow=3, padding=padding)
            grid_pil = transforms.ToPILImage()(grid)
            draw = ImageDraw.Draw(grid_pil)
            font = ImageFont.load_default()
            titles = ["Original", "Masked", "Inpainted"]
            for i, title in enumerate(titles):
                x = i * (sample_image.shape[-1] + padding) + padding
                y = 0
                draw.text((x, y), title, (255, 255, 255), font=font) 
            grid_pil.save(os.path.join(full_path, f"{filename}.png"))

        print(f"Saved inpainting results to {self.args.inpainting_results_dir}")

        print("🎉🎉🎉")


    def spatial_aliasing(self, num_transducers, filename="spatial_aliasing"):
        dataset = load_data(self.args)
        dataset = iter(dataset)
        sample_image = next(dataset).to(device=self.args.device)
        image_size = sample_image.shape[-1]
        mask_width = image_size/(num_transducers*2)
        mask_array = np.zeros((num_transducers, 4),dtype=int)
        for i in range(num_transducers):
            mask_array[i]=[int((2*i+0.5)*mask_width),int((2*i+1.5)*mask_width),0,image_size]
        self.run_inpainting(dataset, mask_array, filename)

    def limited_view(self):
        self.spatial_aliasing(1, "limited_view")



def checkpoint_path_to_config_path(path):

    split_checkpoint_path = path.split("/")

    config_path_li = split_checkpoint_path[:-3]
    config_path_li.append("configs")
    config_path_li.append(split_checkpoint_path[-2] + ".json")

    config_path = "/".join(config_path_li)
    return config_path

if __name__ == '__main__':

    # TODO: Add cmd line args
    # parser = argparse.ArgumentParser(description="VQGAN")
    # parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on.')

    # TODO: Improve we can run different experiments saving to different directories
    # parser.add_argument('--inpainting-results-dir', type=str, default="./results/inpainting_exps")


    # NOTE: Maybe want to abstract into some external json we load so we can run different experiments or just as cmd line args

    parser = argparse.ArgumentParser(description="painting")
    parser.add_argument("--transformer-checkpoint-path", type=str)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--inpainting-results-dir", type=str)
    parser.add_argument("--num-inpainting-images", type=int, default=10)
    parser.add_argument("--num-transducers", type=int, default=10)

    args = parser.parse_args()
    tmp_args = argparse.Namespace(**vars(args))

    transformer_config_path = checkpoint_path_to_config_path(args.transformer_checkpoint_path)

    with open(transformer_config_path, "r") as f:
        json_args = json.load(f)
        args = argparse.Namespace(**json_args)

    args = argparse.Namespace(**json_args)

    # merge args with tmp_args
    args.transformer_checkpoint_path = tmp_args.transformer_checkpoint_path
    args.dataset_path = tmp_args.dataset_path
    args.inpainting_results_dir = tmp_args.inpainting_results_dir
    args.num_inpainting_images = tmp_args.num_inpainting_images
    args.num_transducers = tmp_args.num_transducers
    
    wandb.init(
        project="pat_maskgit_inpainting",
        config=json_args
    )


    # create the inpainting_results_dir if it doesn't exist
    os.makedirs(args.inpainting_results_dir, exist_ok=True)

    # NOTE: Maybe want to abstract as cmd line args
    painting = Painting(args)
    painting.spatial_aliasing(args.num_transducers)
    painting.limited_view()




