from transformer import VQGANTransformer
import torch
import os
import argparse
from torchvision import transforms
from PIL import Image
from utils import load_data, plot_images
from torchvision import utils as vutils
from tqdm import tqdm

class Painting:

    def __init__(self, args):
        self.args = args
        self.model = VQGANTransformer(args).to(device=args.device)
        self.model.load_state_dict(torch.load(args.transformer_checkpoint_path))
    
    def run_inpainting(self):

        # TODO: What images should we inpaint?
        # TODO: We need to figure out a way to encode and pass in the bounded box regions

        # TODO: Custom dataset
        dataset = load_data(self.args)
        # sample_image = next(iter(dataset)).to(device=self.args.device)
        dataset = iter(dataset)
        for i in tqdm(range(self.args.num_inpainting_images)):
            sample_image = next(dataset).to(device=self.args.device)
            masked_image, _ = self.model.create_masked_image(sample_image)
            blended_image, inpainted_image, no_blend_image = self.model.inpainting(sample_image)

            # create a directory for each image
            os.makedirs(os.path.join(self.args.inpainting_results_dir, f"image_{i}"), exist_ok=True)

            full_path = os.path.join(self.args.inpainting_results_dir, f"image_{i}")

            vutils.save_image(masked_image, os.path.join(full_path, f"masked_image.jpg"))
            vutils.save_image(blended_image, os.path.join(full_path, f"blended_image.jpg"))
            vutils.save_image(inpainted_image, os.path.join(full_path, f"inpainted_image.jpg"))
            vutils.save_image(no_blend_image, os.path.join(full_path, f"no_blend_image.jpg"))
            vutils.save_image(sample_image, os.path.join(full_path, f"original_image.jpg"))

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
        "image_size": 256,
        "num_codebook_vectors": 1024,
        "beta": 0.25,
        "image_channels": 3,
        "batch_size": 1,
        "accum_grad": 10,
        "epochs": 1000,
        "learning_rate": 0.0001,
        "n_layers": 24,
        "dim": 768,
        "hidden_dim": 3072,
        "num_image_tokens": 256,
        "checkpoint_path": "./checkpoints/baseline_landscape_3days/vqgan_epoch_240.pt", # vqgan
        "patch_size": 16,

        # NOTE: the following args are custom to this painting task
        "dataset_path": "/groups/mlprojects/pat/landscape/",
        "transformer_checkpoint_path": "./checkpoints/three_day_transformer_landscape_custom_optimizer/transformer_epoch_500.pt", # transformer
        "inpainting_results_dir": "./results/inpainting_exps",
        "device": "cuda",
        "num_inpainting_images": 10,
    }
    args = argparse.Namespace(**args)

    # create the inpainting_results_dir if it doesn't exist
    os.makedirs(args.inpainting_results_dir, exist_ok=True)

    # NOTE: Maybe want to abstract as cmd line args
    painting = Painting(args)
    painting.run_inpainting()






