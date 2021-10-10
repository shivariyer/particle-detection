#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=train_hazy
#SBATCH --output=slurm_haze_%j.out

cd /scratch/ab9738/pollution_img/code/particle-detection/src
python train_haze.py
