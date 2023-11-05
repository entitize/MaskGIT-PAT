#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A mlprojects

#SBATCH --time=3-00:00:00   # walltime, timeout (if script runs longer than specified, it will timeout). Setting it higher results in lower priority on HPC
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem-per-cpu=5G # memory per CPU core
#SBATCH -J "three_day_transformer_landscape_custom_optimizer"   # job name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

# Needed to load miniconda
source ~/.bashrc

module load cuda/11.8

cd ~/MaskGIT-PAT

# NOTE(@kai): the training_vqgan script already uses cuda by default

# CUDA_LAUNCH_BLOCKING=1

python training_transformer.py \
    --run-name three_day_transformer_landscape_custom_optimizer \
    --dataset-path /groups/mlprojects/pat/landscape/ \
    --checkpoint-path ./checkpoints/baseline_landscape_3days/vqgan_epoch_240.pt \
    --batch-size 1 \
    --epochs 1000 \
    --use-custom-optimizer \
