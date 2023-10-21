#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A mlprojects

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem-per-cpu=10G # memory per CPU core
#SBATCH -J "train_vqgan"   # job name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

source ~/.bashrc

module load cuda/11.8

cd ~/MaskGIT-pytorch

# NOTE(@kai): the training_vqgan script already uses cuda by default

python training_vqgan.py --dataset-path /groups/mlprojects/pat/landscape/ --batch-size 6
