"""
Create plots to investigate scaling principles.
"""

import os

import pandas as pd

from utils import LOGS_DIR

import matplotlib.pyplot as plt

# Multi-scale training: Training on multiple sequence lengths
log_folders = [
    # "Uzel2022-NeuralCFC-2023_02_20_22_21",
    # "Uzel2022-NeuralCFC-2023_02_20_22_22",
    # "Uzel2022-NeuralCFC-2023_02_20_22_23",
    # "Uzel2022-NeuralCFC-2023_02_20_22_28",
]

for folder in log_folders:
    path = os.path.join(LOGS_DIR, folder, "loss_curves.csv")
    df = pd.read_csv(path)
