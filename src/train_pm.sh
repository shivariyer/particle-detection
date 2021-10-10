#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=18:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=train_pm
#SBATCH --output=slurm_pm_%j.out

cd /scratch/ab9738/pollution_img/code/particle-detection/src
python train_pm.py
