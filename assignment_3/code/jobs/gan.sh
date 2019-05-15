#!/bin/bash
#SBATCH --job-name=gan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=10000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

source activate pytorch

srun python -u a3_gan_template.py
