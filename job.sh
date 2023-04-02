#!/bin/bash 
#!/bin/bash  
#SBATCH --job-name=train_size  
#SBATCH --output=logs/multirun/train_size.out
#SBATCH -t 12:00:00 
#SBATCH --gres=gpu:2
#SBATCH -n 1
hostname   
#SBATCH --mail-user=kyzhao@mit.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

source activate /om2/user/kyzhao/worm_conda/envs/worm-graph

python main.py  