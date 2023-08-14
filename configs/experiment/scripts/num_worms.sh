#!/bin/bash

#SBATCH --job-name=main
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=64G

#SBATCH --error=configs/experiment/scripts/default_multirun.err
#SBATCH --output=configs/experiment/scripts/default_multirun.out

cd /om2/user/lrvenan/worm-graph
source /om2/user/lrvenan/miniconda/bin/activate worm-graph
python main.py +experiment=num_worms
