#!/bin/bash

#SBATCH --job-name=hyp_search
#SBATCH --output=/om2/user/lrvenan/worm-graph/debugging/leandro/batch_scripts/hyp_search.txt
#SBATCH --gres=gpu:2
#SBATCH --mem=64000
#SBATCH --time=12:00:00

cd /om2/user/lrvenan/worm-graph
source /om2/user/lrvenan/miniconda/bin/activate worm-graph
srun python main.py +experiment=hyperparameter_search
