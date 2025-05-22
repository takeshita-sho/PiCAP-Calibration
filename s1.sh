#!/bin/bash
#
#SBATCH --job-name=p12_ki10
#SBATCH --output=models_DL/p12_ki10.sout
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --account=jgray21_gpu
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4000

ml anaconda3/2024
conda activate picap

python ./train_1.py > models_DL/p12_ki10_log.txt 
srun sleep 5
