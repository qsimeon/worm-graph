#!/bin/bash

#SBATCH --job-name=experiment
#SBATCH --output=/om2/user/lrvenan/worm-graph/debugging/leandro/batch_scripts/experiment_out.txt
#SBATCH --gres=gpu:2
#SBATCH --mem=64000
#SBATCH --time=12:00:00

module load openmind/anaconda/3-2022.05
srun conda activate worm-graph
srun python main.py +submodule=preprocess