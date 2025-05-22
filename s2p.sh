#!/bin/bash
#
#SBATCH --job-name=Dice_pp-BCE
#SBATCH --output=models_DL/tp_12ki10mdice_ppBCE.sout
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

python ./train_2-prot.py > models_DL/tp_12ki10mDice_ppBCE_log.txt
srun sleep 5
