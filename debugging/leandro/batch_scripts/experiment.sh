#!/bin/bash

#SBATCH --job-name=experiment
#SBATCH --output=/om2/user/lrvenan/worm-graph/debugging/leandro/batch_scripts/experiment_out.txt
#SBATCH --gres=gpu:2
#SBATCH --mem=64000
#SBATCH --time=12:00:00

source /om2/user/lrvenan/miniconda/bin/activate worm-graph
srun python main.py +submodule=preprocess
