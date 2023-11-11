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


if __name__ == '__main__':

    # TODO: Add cmd line args
    # parser = argparse.ArgumentParser(description="VQGAN")
    # parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on.')

    # TODO: Improve we can run different experiments saving to different directories
    # parser.add_argument('--inpainting-results-dir', type=str, default="./results/inpainting_exps")


    # NOTE: Maybe want to abstract into some external json we load so we can run different experiments or just as cmd line args
    args = {
        "latent_dim": 256,
        "image_size": 64,
        "num_codebook_vectors": 1024,
        "patch_size":2,
        "beta": 0.25,
        "image_channels": 1,
        "accum_grad": 10,
        "n_layers": 24,
        "dim": 768,
        "hidden_dim": 3072,

        "num_image_tokens": 1024,
        "checkpoint_path": "/central/groups/mlprojects/pat/fanlin/checkpoints/original_pat_only_l2_patch2/vqgan_epoch_20.pt", # vqgan
        "patch_size": 2,

        # NOTE: the following args are custom to this painting task
        "batch_size": 1,
        "dataset_path": "/groups/mlprojects/pat/pat_np/original",
        "transformer_checkpoint_path": "./checkpoints/pat_transformer_48/transformer_current.pt", # transformer
        "inpainting_results_dir": "./results/inpainting_exps2",
        "device": "cuda",
        "num_inpainting_images": 10,
    }
    args = argparse.Namespace(**args)

    # create the inpainting_results_dir if it doesn't exist
    os.makedirs(args.inpainting_results_dir, exist_ok=True)

    # NOTE: Maybe want to abstract as cmd line args
    painting = Painting(args)
    painting.run_inpainting()






