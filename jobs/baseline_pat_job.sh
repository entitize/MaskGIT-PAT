#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A mlprojects

#SBATCH --time=24:00:00   # walltime, timeout (if script runs longer than specified, it will timeout). Setting it higher results in lower priority on HPC
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem-per-cpu=5G # memory per CPU core
#SBATCH -J "pat_np_baseline"   # job name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

# Needed to load miniconda
source ~/.bashrc

module load cuda/11.8

cd ~/MaskGIT-PAT

# NOTE(@kai): the training_vqgan script already uses cuda by default

python training_vqgan.py \
    --dataset-path /groups/mlprojects/pat/pat_norm_crop/train \
    --batch-size 4 \
    --image-size 64 \
    --patch-size 4 \
    --experiment-name pat_patch_size_4_split \
    --save-img-rate 1000 \
    --image-channels 1
