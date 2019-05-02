#!/bin/bash
#SBATCH --job-name=textgen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:30:00
#SBATCH --mem=10000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

source activate pytorch

srun python -um part2.train train --txt_file="part2/assets/t.txt"
