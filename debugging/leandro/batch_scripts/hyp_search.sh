#!/bin/bash

#SBATCH --job-name=search_transformer
#SBATCH --output=/om2/user/lrvenan/worm-graph/debugging/leandro/batch_scripts/search_transformer.txt
#SBATCH --gres=gpu:2
#SBATCH --mem=64000
#SBATCH --time=12:00:00

cd /om2/user/lrvenan/worm-graph
source /om2/user/lrvenan/miniconda/bin/activate worm-graph
srun python main.py +experiment=hyperparameter_transformer
