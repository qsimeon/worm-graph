#!/bin/bash

#SBATCH --job-name=search_lstm
#SBATCH --output=/om2/user/lrvenan/worm-graph/debugging/leandro/batch_scripts/search_lstm.txt
#SBATCH --gres=gpu:4
#SBATCH --mem=32000
#SBATCH --time=24:00:00

cd /om2/user/lrvenan/worm-graph
source /om2/user/lrvenan/miniconda/bin/activate worm-graph
srun python main.py +experiment=hyperparameter_lstm
