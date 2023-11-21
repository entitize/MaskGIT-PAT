#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A mlprojects

#SBATCH --time=1:00:00  # walltime, timeout (if script runs longer than specified, it will timeout). Setting it higher results in lower priority on HPC
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=5G # memory per CPU core
#SBATCH -J "pat_cnn_sa_real"   # job name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

# Needed to load miniconda
source ~/.bashrc
# module avail
module load cuda/11.8
module load gcc/9.2.0
module load clang/16.0.4

export CUDA_VISIBLE_DEVICES=0

cd ~/MaskGIT-PAT

# NOTE(@kai): the training_vqgan script already uses cuda by default

python load_and_predict_cnn.py \
    --result_directory /groups/mlprojects/pat/cnn_spatial_aliasing \
    --dataset-path /groups/mlprojects/pat/pat_norm_crop \
    --image-channels 1 \
    --image-size 64 \
    --model-path wandb/latest-run/files/model-best.h5 \
    --spatial-aliasing
    # --experiment-name cnn_inpainting_check_versions \
