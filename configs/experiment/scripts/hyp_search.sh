#!/bin/bash

#SBATCH --job-name=main
#SBATCH --gres=gpu:1
#SBATCH --partition=yanglab

#SBATCH --error=configs/experiment/scripts/hyp_search.err
#SBATCH --output=configs/experiment/scripts/hyp_search.out

cd /om2/user/lrvenan/worm-graph
source /om2/user/lrvenan/miniconda/bin/activate worm-graph
srun python main.py +experiment=hyperparameter_search
