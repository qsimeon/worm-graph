#!/bin/bash

#SBATCH --job-name=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1440
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --partition=yanglab

#SBATCH --error=configs/experiment/scripts/default_run.err
#SBATCH --output=configs/experiment/scripts/default_run.out

cd /om2/user/lrvenan/worm-graph
source /om2/user/lrvenan/miniconda/bin/activate worm-graph
srun python main.py +experiment=default_run
