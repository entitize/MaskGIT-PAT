#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A mlprojects

#SBATCH --time=1-00:00:00   # walltime, timeout (if script runs longer than specified, it will timeout). Setting it higher results in lower priority on HPC
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=5G # memory per CPU core
#SBATCH -J "reconstruct_job"   # job name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

# Needed to load miniconda
source ~/.bashrc

cd ~/MaskGIT-PAT

python reconstruct.py