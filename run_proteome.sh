#!/bin/bash
#
#SBATCH --job-name=run_proteome
#SBATCH --output=models_DL/run_proteome.sout
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --account=jgray21_gpu
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8000

ml anaconda3/2024
conda activate picap

python ./run_proteome.py > models_DL/run_proteome_log.txt 
srun sleep 5