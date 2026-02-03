#!/bin/bash
#SBATCH --job-name=ahlf-training
#SBATCH --output=slurm_ahlf_training_%j.out
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-batch
#SBATCH --gpus=1
#SBATCH --constraint="GPU_SKU:V100&ARCH:X86"
#SBATCH --account=sci-renard-usi-grabber

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ahlf_training_env

#Specify params in the file directly
python training.py

# Use srun to get an interactive GPU session with the same resources:
# srun -A sci-renard-usi-grabber -p gpu-interactive --gres=gpu:1 --cpus-per-task=4 --mem=16G -C "GPU_SKU:V100&ARCH:X86" --pty bash