"""
Make plots that investigate the scaling properties
of our worm modeling neural networks.
"""

import os

import pandas as pd

import seaborn as sns

import numpy as np

from utils import LOGS_DIR

import matplotlib.pyplot as plt

log_folders = [
    "2023_02_24_18_41-Nichols2017-NeuralCFC",
    # "2023_02_24_15_50-Kaplan2020_Kato2015_Skora2018_Uzel2022-NeuralCFC",
    # "2023_02_24_17_36-Kaplan2020_Kato2015_Skora2018-NeuralCFC",
    # "2023_02_24_18_01-Kaplan2020_Skora2018-NeuralCFC",
]

"""
Effect of number splits of the dataset into train and test blocks. 
The total dataset sizes (i.e. number of train and test samples) is fixed.
"""
plt.figure()
for folder in log_folders:
    # load logs
    log_dir = os.path.join(LOGS_DIR, folder)
    loss_df = pd.read_csv(os.path.join(log_dir, "loss_curves.csv"))
    # plot loss vs samples (sequences)
    sns.lineplot(
        x=np.cumsum(loss_df["num_train_samples"]),
        y="centered_train_losses",
        data=loss_df,
    )
    # sns.lineplot(
    #     x=np.cumsum(loss_df["num_test_samples"]),
    #     y="centered_test_losses",
    #     data=loss_df,
    #     label="test",
    # )
    # plt.legend()
    plt.title("")
    plt.xlabel("Samples (# sequences)")
    plt.ylabel("Loss - Baseline")
# plt.savefig(os.path.join(log_dir, "sample_vs_loss_curves.png"))
plt.show()
