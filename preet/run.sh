#!/bin/bash
# Job name:
#SBATCH --job-name=temporal
# Partition:
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=phd
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 ## Define number of GPUs
#SBATCH --output=/csehome/p24cs0005/ppmf/run_new_new.out
#SBATCH --error=/csehome/p24cs0005/ppmf/run_new_new.out
#SBATCH --nodelist=cn01

module load gcc/11.4.0-gcc-12.3.0-73jjveq
module load cuda/11.8.0-gcc-12.3.0-4pg4hmh

python train.py