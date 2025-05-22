#!/bin/bash
#
#SBATCH --job-name=tc_12kd40
#SBATCH --output=models_DL/tc_12kd40.sout
#
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --account=jgray21_gpu
#SBATCH --time=18:00:00
#SBATCH --mem-per-cpu=4000

module unload python
module load cuda/11.1.0
module load python/3.9.0
source ~/my_env/bin/activate

python ./train_2-carb.py > models_DL/tc_12kd40_log.txt
srun sleep 5
