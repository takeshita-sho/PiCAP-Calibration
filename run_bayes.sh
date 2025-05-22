#!/bin/bash

#SBATCH --job-name="run_bayes"
#SBATCH --output="models_DL/run_bayes_%j.sout"
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --account=jgray21_gpu
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8000
n="5"
ml anaconda3/2024
conda activate picap

python ./run_bayes.py ${n} > "models_DL/run_bayes_log_${n}.txt" 
srun sleep 5