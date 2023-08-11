#!/bin/bash

cd /om2/user/lrvenan/worm-graph
source /om2/user/lrvenan/miniconda/bin/activate worm-graph
srun python main.py +experiment=default_run
