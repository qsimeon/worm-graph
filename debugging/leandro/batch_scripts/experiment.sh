#!/bin/bash

SBATCH --job-name=experiment
SBATCH --output=experiment_out.txt
SBATCH --gres=gpu:A100:2
SBATCH --mem=64000
SBATCH --time=12:00:00

#SBATCH --job-name=my_job
#SBATCH --output=res.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

module add openmind/anaconda/3-2022.05
srun hostname
srun conda activate worm-graph
srun python main.py +experiment=default_run