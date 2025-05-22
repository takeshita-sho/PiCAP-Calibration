#!/bin/bash
#
#SBATCH --job-name=iso_bar
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=shared
#SBATCH --account=jgray21
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=4000

ml anaconda3/2024
conda activate myenv

python ./iso_bar.py
srun sleep 5