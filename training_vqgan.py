import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from torchvision import transforms
from discriminator import Discriminator
from lpips import LPIPS
from utils import load_data, weights_init
from vqgan import VQGAN
import datetime
import json
from torch.utils.tensorboard import SummaryWriter

class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)
        self.start_epoch = 0

        self.logger = SummaryWriter(f"./runs/{args.experiment_name}")

        self.prepare_training()
        if args.resume_exp:
            self.resume_training()

        self.train(args)

    def prepare_training(self):
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("configs", exist_ok=True)

        os.makedirs(os.path.join("results", args.experiment_name), exist_ok=True)
        os.makedirs(os.path.join("checkpoints", args.experiment_name), exist_ok=True)
        with open(os.path.join("configs", args.experiment_name + ".json"), "w") as f:
            args_dict = vars(args)
            args_dict["time"] = str(datetime.datetime.now())
            json.dump(args_dict, f, indent=4)


    def resume_training(self):
        checkpoints = os.listdir(os.path.join("checkpoints", args.experiment_name))
        if len(checkpoints) > 0:
            checkpoints.sort()
            newest_epoch = checkpoints[-1][:-3].split('_')[-1]
            newest_epoch = int(newest_epoch)
            for checkpoint in checkpoints:
                epoch = checkpoint[:-3].split('_')[-1]
                epoch = int(epoch)
                if epoch > newest_epoch:
                    newest_epoch = epoch
            checkpoint = torch.load(os.path.join("checkpoints", args.experiment_name, f"vqgan_epoch_{newest_epoch}.pt"))
            print(f"Resuming from epoch: {newest_epoch}")
            self.vqgan.load_state_dict(checkpoint["vqgan"])
            self.discriminator.load_state_dict(checkpoint["discriminator"])
            self.opt_vq.load_state_dict(checkpoint["opt_vq"])
            self.opt_disc.load_state_dict(checkpoint["opt_disc"])
            self.start_epoch = newest_epoch + 1
        else:
            print("Find no checkpoint!")
                    
    
    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(list(self.vqgan.encoder.parameters()) +
                                  list(self.vqgan.decoder.parameters()) +
                                  list(self.vqgan.codebook.parameters()) +
                                  list(self.vqgan.quant_conv.parameters()) +
                                  list(self.vqgan.post_quant_conv.parameters()),
                                  lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))
        return opt_vq, opt_disc

    def train(self, args):
        train_dataset = load_data(args)
        steps_one_epoch = len(train_dataset)
        for epoch in range(self.start_epoch, args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(device=args.device)

                    decoded_images, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch * steps_one_epoch + i,
                                                          threshold=args.disc_start)
                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    nll_loss = args.perceptual_loss_factor * perceptual_loss + args.l2_loss_factor * rec_loss
                    nll_losss = nll_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(nll_losss, g_loss)
                    loss_vq = nll_losss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    loss_gan = disc_factor * .5 * (d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    loss_vq.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    loss_gan.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    if i % args.save_img_rate == 0:
                        with torch.no_grad():
                            color_bar = torch.linspace(0, 1, args.image_size).unsqueeze(1).expand(1, args.image_channels, args.image_size, args.image_size)
                            color_bar = color_bar.to(device=args.device)
                            both = torch.cat((imgs.add(1).mul(0.5)[:4], color_bar, decoded_images.add(1).mul(0.5)[:4], color_bar))
                            vutils.save_image(both, os.path.join("results", args.experiment_name, f"{epoch}_{i}.jpg"), nrow=5)

                            # save image to tensorboard
                            grid = vutils.make_grid(both, nrow=5)
                            self.logger.add_image(f"train_images_{i}", grid, (epoch * steps_one_epoch) + i)
                            self.logger.add_scalar("Save Rec Loss 1", np.round(rec_loss[0].mean().cpu().detach().numpy().item(), 5), (epoch * steps_one_epoch) + i)
                            self.logger.add_scalar("Save Rec Loss 2", np.round(rec_loss[1].mean().cpu().detach().numpy().item(), 5), (epoch * steps_one_epoch) + i)
                            self.logger.add_scalar("Save Rec Loss 3", np.round(rec_loss[2].mean().cpu().detach().numpy().item(), 5), (epoch * steps_one_epoch) + i)
                            self.logger.add_scalar("Save Rec Loss 4", np.round(rec_loss[3].mean().cpu().detach().numpy().item(), 5), (epoch * steps_one_epoch) + i)
                            

                    pbar.set_postfix(VQ_Loss=np.round(loss_vq.cpu().detach().numpy().item(), 5),
                                     GAN_Loss=np.round(loss_gan.cpu().detach().numpy().item(), 3))
                    pbar.update(0)

                    self.logger.add_scalar("VQ Loss", np.round(loss_vq.cpu().detach().numpy().item(), 5), (epoch * steps_one_epoch) + i)
                    self.logger.add_scalar("Rec Loss", np.round(rec_loss.mean().cpu().detach().numpy().item(), 5), (epoch * steps_one_epoch) + i)
                    self.logger.add_scalar("Perceptual Loss", np.round(perceptual_loss.mean().cpu().detach().numpy().item(), 5), (epoch * steps_one_epoch) + i)
                    self.logger.add_scalar("Codebook Loss", np.round(q_loss.cpu().detach().numpy().item(), 5), (epoch * steps_one_epoch) + i)
                    self.logger.add_scalar("G Loss", np.round(g_loss.mean().cpu().detach().numpy().item(), 5), (epoch * steps_one_epoch) + i)
                    self.logger.add_scalar("Disc Loss", np.round(loss_gan.cpu().detach().numpy().item(), 3), (epoch * steps_one_epoch) + i)
                checkpoint = {
                    "vqgan":self.vqgan.state_dict(),
                    "discriminator":self.discriminator.state_dict(),
                    "opt_vq":self.opt_vq.state_dict(),
                    "opt_disc":self.opt_disc.state_dict(),
                }
                torch.save(checkpoint, os.path.join("checkpoints", args.experiment_name, f"vqgan_epoch_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--patch-size', type=int, default=16, help='Height and width of a patch (default: 16)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=6, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    parser.add_argument('--save-img-rate', type=int, default=1000, help='How often to save images (default: 1000). In units of steps')

    parser.add_argument('--experiment-name', type=str, help='Name of experiment. This will be used to create a folder in results/ and checkpoints/')
    parser.add_argument('--resume-exp', action="store_true", help='Name of experiment. This will be used to create a folder in results/ and checkpoints/')

    args = parser.parse_args()

    if args.resume_exp:
        print("Resuming experiment: ", args.experiment_name)
    else:
        # User should specify unique experiment name
        i = 1
        original_experiment_name = args.experiment_name
        while os.path.exists(os.path.join("checkpoints", args.experiment_name)):
            args.experiment_name = original_experiment_name + "_" + str(i)
            i += 1
        if i > 1:
            print("Experiment name already exists. Changing experiment name to: ", args.experiment_name)

        print("Running experiment: ", args.experiment_name)

    train_vqgan = TrainVQGAN(args)
