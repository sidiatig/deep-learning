#!/bin/bash
#SBATCH --job-name=rnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:05:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

source activate pytorch

srun python -u train.py --model_type="LSTM" --input_length=5
