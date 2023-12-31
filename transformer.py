import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from bidirectional_transformer import BidirectionalTransformer
from vqgan import VQGAN
import matplotlib.pyplot as plt
from torchvision import utils as vutils

_CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to("cuda")

class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_image_tokens = args.num_image_tokens
        self.sos_token = args.num_codebook_vectors + 1
        self.mask_token_id = args.num_codebook_vectors
        self.choice_temperature = 4.5

        self.gamma = self.gamma_func("cosine")

        # self.transformer = BidirectionalTransformer(
        #                         patch_size=8, embed_dim=args.dim, depth=args.n_layers, num_heads=12, mlp_ratio=4, qkv_bias=True,
        #                         norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192+1)
        self.transformer = BidirectionalTransformer(args)
        self.vqgan = self.load_vqgan(args)
        print(f"Transformer parameters: {sum([p.numel() for p in self.transformer.parameters()])}")

    def load_checkpoint(self, epoch):
        self.load_state_dict(torch.load(os.path.join("checkpoints", f"transformer_epoch_{epoch}.pt")))
        print("Check!")

    @staticmethod
    def load_vqgan(args):
        model = VQGAN(args)
        # model.load_checkpoint(args.checkpoint_path)
        if "vqgan" in torch.load(args.checkpoint_path):
            model.load_state_dict(torch.load(args.checkpoint_path)["vqgan"])
        else:
            model.load_state_dict(torch.load(args.checkpoint_path))            
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        # quant_z, _, (_, _, indices) = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    def forward(self, x):
        # _, z_indices = self.encode_to_z(x)
        #
        # r = np.random.uniform()
        # mask = torch.bernoulli(r * torch.ones(z_indices.shape[-1], device=z_indices.device))
        # mask = mask.round().bool()
        #
        # target = z_indices[:, mask]
        #
        # logits = self.transformer(z_indices, mask)

        _, z_indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0], 1, dtype=torch.long, device=z_indices.device) * self.sos_token

        r = math.floor(self.gamma(np.random.uniform()) * z_indices.shape[1])
        sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
        mask.scatter_(dim=1, index=sample, value=True)

        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)
        a_indices = mask * z_indices + (~mask) * masked_indices

        a_indices = torch.cat((sos_tokens, a_indices), dim=1)

        target = torch.cat((sos_tokens, z_indices), dim=1)

        logits = self.transformer(a_indices)

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        if k == 0:
            out[:, :] = self.sos_token
        else:
            out[out < v[..., [-1]]] = self.sos_token
        return out

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    def create_input_tokens_normal(self, num, label=None):
        # label_tokens = label * torch.ones([num, 1])
        # Shift the label by codebook_size
        # label_tokens = label_tokens + self.vqgan.codebook.num_codebook_vectors
        # Create blank masked tokens
        blank_tokens = torch.ones((num, self.num_image_tokens), device=self.args.device)
        masked_tokens = self.mask_token_id * blank_tokens
        # Concatenate the two as input_tokens
        # input_tokens = torch.concat([label_tokens, masked_tokens], dim=-1)
        # return input_tokens.to(torch.int32)
        return masked_tokens.to(torch.int64)

    def tokens_to_logits(self, seq):
        logits = self.transformer(seq)
        # logits = logits[..., :self.vqgan.codebook.num_codebook_vectors]  # why is maskgit returning [8, 257, 2025]?
        return logits

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(self.args.device)
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking

    @torch.no_grad()
    def sample_good(self, inputs=None, num=1, T=11, mode="cosine"):
        self.transformer.eval()


        N = self.num_image_tokens
        if inputs is None:
            inputs = self.create_input_tokens_normal(num)
        else:
            inputs = torch.hstack(
                (inputs, torch.zeros((inputs.shape[0], N - inputs.shape[1]), device=self.args.device, dtype=torch.int).fill_(self.mask_token_id)))

        sos_tokens = torch.ones(inputs.shape[0], 1, dtype=torch.long, device=inputs.device) * self.sos_token
        inputs = torch.cat((sos_tokens, inputs), dim=1)

        unknown_number_in_the_beginning = torch.sum(inputs == self.mask_token_id, dim=-1)
        gamma = self.gamma_func(mode)
        cur_ids = inputs  # [8, 257]
        # for t in range(T):
        t = 0
        while (cur_ids == self.mask_token_id).any():
            logits = self.tokens_to_logits(cur_ids)  # call transformer to get predictions [8, 257, 1024]
            sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()

            unknown_map = (cur_ids == self.mask_token_id)  # which tokens need to be sampled -> bool [8, 257]
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)  # replace all -1 with their samples and leave the others untouched [8, 257]

            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            probs = F.softmax(logits, dim=-1)  # convert logits into probs [8, 257, 1024]
            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1)  # get probability for selected tokens in categorical call, also for already sampled ones [8, 257]

            selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)  # ignore tokens which are already sampled [8, 257]

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....]
            mask_len = torch.maximum(torch.zeros_like(mask_len), torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True)-1, mask_len))  # add -1 later when conditioning and also ones_like. Zeroes just because we have no cond token
            # max(1, min(how many unknown tokens, how many tokens we want to sample))

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature * (1. - ratio))
            # Masks tokens with lower confidence.
            cur_ids = torch.where(masking, self.mask_token_id, sampled_ids)
            # print((cur_ids == 8192).count_nonzero())
            t += 1

        self.transformer.train()
        return cur_ids[:, 1:]

    @torch.no_grad()
    def log_images(self, x, mode="cosine"):
        log = dict()

        _, z_indices = self.encode_to_z(x)

        # create new sample
        index_sample = self.sample_good(mode=mode)
        x_new = self.indices_to_image(index_sample)

        # create a "half" sample
        z_start_indices = z_indices[:, :z_indices.shape[1] // 2]
        half_index_sample = self.sample_good(z_start_indices, mode=mode)
        x_sample = self.indices_to_image(half_index_sample)

        # create reconstruction
        x_rec = self.indices_to_image(z_indices)

        # create inpainting middle region
        masked_image, _ = self.create_masked_image(x, 10, 10, 10)
        _, _, no_blend_image = self.inpainting(x, 10, 10, 10)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = x_sample
        log["new_sample"] = x_new
        log["masked_img"] = masked_image
        log["no_blend_image"] = no_blend_image
        return log, torch.concat((x, x_rec, x_sample, x_new, masked_image, no_blend_image, x))
    

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
        plt.savefig(full_path)
        plt.close(fig)

    @torch.no_grad() 
    def log_custom_images(self, x, full_path, mode="cosine"):

        index_map, masked_image, inpainted_image = self.custom_inpainting(x)
            
        self.plot_index_map(index_map, full_path)

        vutils.save_image(torch.cat((x, masked_image, inpainted_image)).add(1).mul(0.5), os.path.join(full_path, f"inpainted_image.jpg"), nrow=4)

    def indices_to_image(self, indices):
        indices = indices.clone()
        p1 = p2 = int(math.sqrt(self.args.num_image_tokens))
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, -1)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image
    

    @staticmethod
    def create_masked_image(image: torch.Tensor, mask_array):
        mask = torch.ones_like(image)
        for row in mask_array:
            x_min, x_max, y_min, y_max = row
            mask[:, :, y_min:y_max, x_min:x_max] = 0
        average_color = image.mean()
        masked_image = image * mask
        if (image.shape[1] == 3):
            average_color = image.mean(dim=(2, 3), keepdim=True)
            masked_image = torch.where(masked_image == 0, average_color, masked_image)
        masked_image = torch.where(masked_image == 0, average_color, masked_image)
        return masked_image, mask
    
    @torch.no_grad()
    def custom_inpainting(self, x, mask_array):
        # mask images first to prevent leakage
        _, _, H, W = x.shape
        p = int(math.sqrt(self.args.num_image_tokens))
        masked_image, mask = self.create_masked_image(x, mask_array)

        # encode to z_indices
        q = H//p
        _, z_indices = self.encode_to_z(masked_image)
        _, original_z_indices = self.encode_to_z(x)
        z_indices = z_indices.reshape(z_indices.shape[0], p, p)

        mask_tokens = torch.nn.AvgPool2d(q)(mask)
        z_indices[mask_tokens[0,0].unsqueeze(0) < 0.4] = self.mask_token_id
     
        z_indices = z_indices.reshape(z_indices.shape[0], -1).to(torch.int64).to(self.args.device)
                                                
        inpainted_indices = self.sample_good(z_indices)
        inpainted_image = self.indices_to_image(inpainted_indices)
        inpainted_image = torch.where(x*mask == 0, inpainted_image, x)


        index_map = {
            "original": original_z_indices.cpu(),
            "masked": z_indices.cpu(),
            "inpainted": inpainted_indices.cpu(),
        }

        return index_map, masked_image, inpainted_image


