#!/bin/bash
#
#SBATCH --job-name=run_picap
#SBATCH --output=models_DL/run_picap.sout
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --account=jgray21_gpu
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4000

ml anaconda3/2024
conda activate picap

python ./run_picap.py > models_DL/run_picap_log.txt 
srun sleep 5