#!/bin/bash
#
#SBATCH --job-name=iso_fit
#SBATCH --output=models_DL/iso_fit.sout
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --account=jgray21_gpu
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=4000

ml anaconda3/2024
conda activate picap

python ./fit_isotonic.py > models_DL/iso_fit_log.txt 
srun sleep 5