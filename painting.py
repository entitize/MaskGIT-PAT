from transformer import VQGANTransformer
import torch
import numpy as np
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
from utils import load_data, plot_images
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import utils
# import wandb
import json

class Painting:

    def __init__(self, args):
        self.args = args
        self.model = VQGANTransformer(args).to(device=args.device)
        # get number of parameters in class VQGANTransformer(nn.Module):
        print(f"Number of parameters in model: {sum(p.numel() for p in self.model.parameters())}")
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

            if (sample_image_squeezed.shape[0] == 1):
                np.save(os.path.join(full_path, f"{filename}_original.npy"), sample_image_squeezed.clone().cpu().numpy())
                np.save(os.path.join(full_path, f"{filename}_masked.npy"), masked_image_squeezed.clone().cpu().numpy())
                np.save(os.path.join(full_path, f"{filename}_inpainted.npy"), inpainted_image_squeezed.clone().cpu().numpy())
            
            result_path = os.path.join(full_path, f"{filename}.png")
            utils.display_results(mask_array, sample_image, masked_image, inpainted_image, result_path)

        print(f"Saved inpainting results to {self.args.inpainting_results_dir}")

        print("🎉🎉🎉")


    def spatial_aliasing(self, num_transducers, filename="spatial_aliasing" ):
        dataset = load_data(self.args)
        dataset = iter(dataset)
        sample_image = next(dataset).to(device=self.args.device)
        image_size = sample_image.shape[-1]
        mask_width = image_size/(num_transducers*2)
        mask_array = np.zeros((num_transducers, 4),dtype=int)
        for i in range(num_transducers):
            mask_array[i]=[int((2*i+1)*mask_width),int((2*i+2)*mask_width),0,image_size]
        self.run_inpainting(dataset, mask_array, filename)


    def limited_view(self, filename="limited_view"):
        dataset = load_data(self.args)
        dataset = iter(dataset)
        sample_image = next(dataset).to(device=self.args.device)
        image_size = sample_image.shape[-1]
        mask_width = image_size/2
        mask_array = np.zeros((2, 4),dtype=int)
        for i in range(2):
            mask_array[i]=[int((2*i+0.5)*mask_width),int((2*i+1.5)*mask_width),0,image_size]
        self.run_inpainting(dataset, mask_array, filename)


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
    
    # wandb.init(
    #     project="pat_maskgit_inpainting",
    #     config=json_args
    # )


    # create the inpainting_results_dir if it doesn't exist
    os.makedirs(args.inpainting_results_dir, exist_ok=True)

    # NOTE: Maybe want to abstract as cmd line args
    painting = Painting(args)
    painting.spatial_aliasing(args.num_transducers)
    painting.limited_view()