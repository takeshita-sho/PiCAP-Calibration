#!/bin/bash
#
#SBATCH --job-name=bayes2
#SBATCH --output=models_DL/bayes2.sout
#
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --account=jgray21_gpu
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=16000

ml anaconda3/2024
conda activate picap

python ./train_2-bayes.py > models_DL/bayes2_log.txt
srun sleep 5