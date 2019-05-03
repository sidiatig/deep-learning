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

srun python -um part3.train train --midi_file="part3/assets/chno0901.mid" --train_steps=600000
