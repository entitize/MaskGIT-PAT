#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A mlprojects

#SBATCH --time=00:10:00   # walltime, timeout (if script runs longer than specified, it will timeout). Setting it higher results in lower priority on HPC
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=5G # memory per CPU core
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH -J "baseline_inpainting"   # job name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

# Needed to load miniconda
source ~/.bashrc

module load cuda/11.8

cd ~/MaskGIT-PAT

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1


python painting.py \
    --transformer-checkpoint-path /groups/mlprojects/pat/kai/checkpoints/pat_transformer_62/transformer_current.pt \
    --dataset-path /groups/mlprojects/pat/pat_np/original \
    --inpainting-results-dir ./results/inpainting_exps \
    --num-transducers 9 \
    --num-inpainting-images 20
