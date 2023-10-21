# MaskGIT-pytorch

Commands to run to setup HPC
```
cd ~
git clone https://github.com/entitize/MaskGIT-PAT/

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

source ~/.bashrc
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy
pip install albumentations
pip install matplotlib
pip install tensorboard
```

To run baseline
```
cd jobs
sbatch baseline_landscape_job.sh
```

To monitor in realtime
```
cd jobs
tail -f LOGFILE.out
```

Tensorboard monitoring
```
tensorboard --logdir=.
```

Pytorch implementation of MaskGIT: Masked Generative Image Transformer (https://arxiv.org/pdf/2202.04200.pdf)
<p align="center">
<img width="718" alt="results" src="https://user-images.githubusercontent.com/61938694/154553460-3eb2b55e-e313-4100-bc5e-b9d8c4dd8cd7.png">
</p>

#### Note: this is work in progress + the official implementation can be found at https://github.com/google-research/maskgit


MaskGIT is an extension to the VQGAN paper which improves the second stage transformer part (and leaves the first stage untouched). It switches the unidirectional transformer for a bidirectional transformer. The (second stage) training is pretty similar to BERT by randomly masking out tokens and trying to predict these using the bidirectional transformer (the original work used a GPT architecture randomly replaced tokens by other tokens). Different from BERT, the percentage for the masking is not fixed and uniformly distributed between 0 and 1 for each batch. Furhtermore, a new inference algorithm is suggested in which we start off by a completely masked-out image and then iteratively sample vectors where the model has a high confidence.

If you are only interested in the part of the code that comes from this paper check out [transformer.py](https://github.com/dome272/MaskGIT-pytorch/blob/main/transformer.py).

## Results
(Training on https://www.kaggle.com/arnaud58/landscape-pictures, epochs=1000, bs=100)
<p>
  <img src="https://user-images.githubusercontent.com/61938694/163984267-4e22fd7b-512b-43b3-8fcf-002595e066e7.png" width="200"/>
  <img src="https://user-images.githubusercontent.com/61938694/163984994-95c44898-3734-4438-8c6b-6c1c1cc86920.png" width="200"/>
  <img src="https://user-images.githubusercontent.com/61938694/163985169-07cd7fb8-5517-41e3-83b2-7f2c99e3da8d.png" width="200"/>
  <img src="https://user-images.githubusercontent.com/61938694/163985493-0beb72bb-7e8a-4c9d-91f7-301e25ef42e6.png" width="200"/>
</p>
Note: The training only encompasses about 3% of data of what the original paper trained on. (8.000 * 1.000 / 1.000.000 * 300 = 0.026)
So longer training probably results in better outcomes. See also Issue https://github.com/dome272/MaskGIT-pytorch/issues/6

## Run the code
The code is ready for training both the VQGAN and the Bidirectional Transformer and can also be used for inference

```python training_vqgan.py```

```python training_transformer.py```

(Make sure to edit the path for the dataset etc.)

## TODO
- [x] Implement the gamma functions
- [ ] Implement functions for image editing tasks: inpainting, extrapolation, image manipulation
  - started working on inpainting function. [transformer.py](https://github.com/dome272/MaskGIT-pytorch/blob/main/transformer.py#L152)
- [ ] Tune hyperparameters
- [x] Provide visual results
- [ ] Pre-Norm instead of Post-Norm

