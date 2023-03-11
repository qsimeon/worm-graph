"""
Investigate the interleaved splits of the dataset into train and test blocks. 
The total dataset sizes (i.e. number of train and test samples) is fixed.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import LOGS_DIR


log_folders = [
    "2023_02_24_19_55-Nichols2017-NeuralCFC",
    # "2023_02_24_18_01-Kaplan2020_Skora2018-NeuralCFC",
]

if __name__ == "__main__":
    plt.figure()
    for folder in log_folders:
        # load logs
        log_dir = os.path.join(LOGS_DIR, folder)
        loss_df = pd.read_csv(os.path.join(log_dir, "loss_curves.csv"))
        # plot loss vs samples (sequences)
        sns.lineplot(
            data=loss_df,
            x="epochs",
            y=np.cumsum(loss_df["num_train_samples"]),
            label="train",
        )
        sns.lineplot(
            data=loss_df,
            x="epochs",
            y=np.cumsum(loss_df["num_test_samples"]),
            label="test samples",
        )
        plt.title("Cumulative Number of Sampled Sequences versus Epochs")
        plt.xlabel("Epochs (# worms)")
        plt.ylabel("Cumulative number of samples (# seqeuences)")
        plt.savefig(os.path.join(log_dir, "cumulative_samples.png"))
    plt.close()
