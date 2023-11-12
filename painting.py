from transformer import VQGANTransformer
import torch
import os
import argparse
from torchvision import transforms
from PIL import Image
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

    def plot_index_map(self, idxs_map, full_path):
        # Create a new figure
        fig = plt.figure(figsize=(10, 10))

        # Calculate the number of rows and columns for the subplot grid
        num_images = len(idxs_map)
        num_cols = round(math.sqrt(num_images))
        num_rows = num_images // num_cols
        if num_images % num_cols: num_rows += 1

        for i, (idx_name, idx_grid) in enumerate(idxs_map.items()):
            # Normalize the indices and convert to RGB
            normalized_indices = (idx_grid / torch.max(idx_grid))
            image = plt.get_cmap('viridis')(normalized_indices)[:, :, :3]
            image = image.reshape(32, 32, 3)

            # Add a subplot for this image
            ax = fig.add_subplot(num_rows, num_cols, i+1)
            ax.imshow(image)
            ax.set_title(idx_name)
            ax.axis('off')

        # Save the figure
        plt.savefig(os.path.join(full_path, f'index_map.jpg'))
        plt.close(fig)
    
    def run_inpainting(self):

        # TODO: What images should we inpaint?
        # TODO: We need to figure out a way to encode and pass in the bounded box regions

        # TODO: Custom dataset
        dataset = load_data(self.args)
        # sample_image = next(iter(dataset)).to(device=self.args.device)
        dataset = iter(dataset)
        for i in tqdm(range(self.args.num_inpainting_images)):
            sample_image = next(dataset).to(device=self.args.device)

            # custom inpainting (not using provided inpainting function), no blending
            idxs_map, masked_image, inpainted_image = self.model.custom_inpainting(sample_image)

            # create a directory for each image
            os.makedirs(os.path.join(self.args.inpainting_results_dir, f"image_{i}"), exist_ok=True)
            full_path = os.path.join(self.args.inpainting_results_dir, f"image_{i}")

            self.plot_index_map(idxs_map, full_path)

            vutils.save_image(sample_image, os.path.join(full_path, f"original_image.jpg"))
            vutils.save_image(masked_image, os.path.join(full_path, f"masked_image.jpg"))
            vutils.save_image(inpainted_image, os.path.join(full_path, f"inpainted_image.jpg"))
            

        print(f"Saved inpainting results to {self.args.inpainting_results_dir}")

        print("🎉🎉🎉")


    def run_outpainting(self):
        # TODO maybe don't need seperate function since we do the same thing as inpainting
        pass


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
    
    wandb.init(
        project="pat_maskgit_inpainting",
        
        config=json_args
    )


    # create the inpainting_results_dir if it doesn't exist
    os.makedirs(args.inpainting_results_dir, exist_ok=True)

    # NOTE: Maybe want to abstract as cmd line args
    painting = Painting(args)
    painting.run_inpainting()






