import os
import torch
import random
import numpy as np
import pandas as pd
import torch.multiprocessing

# set the start method for multiprocessing
torch.multiprocessing.set_start_method("spawn", force=True)

# set some environment variables
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["OC_CAUSE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

NUM_NEURONS = 302

RAW_DATA_URL = "https://www.dropbox.com/s/45yqpvtsncx4095/raw_data.zip?dl=1"

RAW_FILES = [
    "GHermChem_Edges.csv",
    "GHermChem_Nodes.csv",
    "GHermElec_Sym_Edges.csv",
    "GHermElec_Sym_Nodes.csv",
    "LowResAtlasWithHighResHeadsAndTails.csv",
    "neurons_302.txt",
]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# get GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set of C. elegans datasets we have processed
VALID_DATASETS = {
    # real worm datasets
    "Uzel2022",
    "Skora2018",
    "Nichols2017",
    "Nguyen2017",  # no named neurons. no time vectors. DON'T use!
    "Kato2015",
    "Kaplan2020",
    "Leifer2023",
    "Flavell2023",  # something wrong with worm0.
    # test datasets
    "sine",
    "sine_seq",
    "sine_seq_noise",
    "sine_noise",
    "sum_sine",
    "sum_sine_noise",
}

# List of all 302 hermaphrodite neurons
NEURONS_302 = sorted(
    pd.read_csv(
        os.path.join(ROOT_DIR, "data", "raw", "neurons_302.txt"),
        sep=" ",
        header=None,
        names=["neuron"],
    ).neuron
)


def init_random_seeds(seed=0):
    """
    Initialize random seeds for numpy, torch, and random.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return None
