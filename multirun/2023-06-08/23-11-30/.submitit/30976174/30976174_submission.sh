#!/bin/bash

# Parameters
#SBATCH --error=/net/vast-storage/scratch/vast/yanglab/qsimeon/worm-graph/multirun/2023-06-08/23-11-30/.submitit/%j/%j_0_log.err
#SBATCH --job-name=my_script
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/net/vast-storage/scratch/vast/yanglab/qsimeon/worm-graph/multirun/2023-06-08/23-11-30/.submitit/%j/%j_0_log.out
#SBATCH --signal=USR2@120
#SBATCH --time=60
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /net/vast-storage/scratch/vast/yanglab/qsimeon/worm-graph/multirun/2023-06-08/23-11-30/.submitit/%j/%j_%t_log.out --error /net/vast-storage/scratch/vast/yanglab/qsimeon/worm-graph/multirun/2023-06-08/23-11-30/.submitit/%j/%j_%t_log.err /om2/user/qsimeon/miniconda/envs/worm-graph/bin/python -u -m submitit.core._submit /net/vast-storage/scratch/vast/yanglab/qsimeon/worm-graph/multirun/2023-06-08/23-11-30/.submitit/%j
