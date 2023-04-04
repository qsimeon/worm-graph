#!/bin/bash 
srun -n 1 --gres=gpu:1 --constraint=any-gpu -t 12:00:00 --mem=16G --pty bash