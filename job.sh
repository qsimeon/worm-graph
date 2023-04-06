#!/bin/bash                      
#SBATCH -t 6:00:00 
#SBATCH mem=256G
#SBATCH --gres=gpu:2
#SBATCH -n 6    
hostname   
#SBATCH --mail-user=kyzhao@mit.edu    
python main.py  
