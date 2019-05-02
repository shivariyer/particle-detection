#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=train_hazy
#SBATCH --output=slurm_hazy_%j.out

module load python/intel/2.7.12
module load opencv/intel/2.4.13.2
module load pytorch/python2.7/0.3.0_4
module load torchvision/0.1.8

#export PYTHONPATH=$PYTHONPATH:/scratch/yj627/pm/src
cd /scratch/yj627/pm/src
python train_hazy.py
