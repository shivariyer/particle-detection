#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=train_pm
#SBATCH --output=slurm_pm_%j.out

module load python/intel/2.7.12
module load opencv/intel/2.4.13.2
module load pytorch/python2.7/0.3.0_4
module load torchvision/0.1.8

#export PYTHONPATH=$PYTHONPATH:/scratch/yj627/pm/src
cd /scratch/yj627/pm/src
python train_pm.py
