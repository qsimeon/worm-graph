import os
import torch
import pandas as pd

# defines `worm_graph` as the root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# get GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set of C. elegans datasets we have processed
VALID_DATASETS = {
    "Nichols2017",
    "Nguyen2017",
    "Skora2018",
    "Kaplan2020",
    "Uzel2022",
    "Kato2015",
}

# list of all 302 hermaphrodite neurons
NEURONS_302 = sorted(
        pd.read_csv(
            os.path.join(ROOT_DIR, "data", "raw", "neurons_302.txt"),
            sep=" ",
            header=None,
            names=["neuron"],
        ).neuron
    )