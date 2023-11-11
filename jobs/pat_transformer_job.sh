#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A mlprojects

#SBATCH --time=00:10:00   # walltime, timeout (if script runs longer than specified, it will timeout). Setting it higher results in lower priority on HPC
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem-per-cpu=5G # memory per CPU core
#SBATCH -J "pat_transformer"   # job name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

# Needed to load miniconda
source ~/.bashrc

module load cuda/11.8

cd ~/MaskGIT-PAT

python training_transformer.py \
    --run-name pat_transformer \
    --dataset-path /groups/mlprojects/pat/pat_np/original \
    --checkpoint-path /central/groups/mlprojects/pat/fanlin/checkpoints/original_pat_only_l2_patch2/vqgan_epoch_20.pt \
    --batch-size 1 \
    --epochs 100000 \
    --patch-size 2 \
    --image-channels 1 \
    --image-size 64 \
    --num-image-tokens 1024 \
    --num-train-samples 10 \
    --disable-log-images