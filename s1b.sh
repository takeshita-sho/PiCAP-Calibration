#!/bin/bash
#
#SBATCH --job-name=bayes
#SBATCH --output=models_DL/bayes.sout
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --account=jgray21_gpu
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=8000

ml anaconda3/2024
conda activate picap

python ./train_1_bayes.py > models_DL/bayes_log.txt 
srun sleep 5