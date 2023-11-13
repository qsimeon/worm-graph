import os
import torch
import random
import warnings
import numpy as np
import pandas as pd
import torch.multiprocessing

# Ignore all warnings
warnings.filterwarnings(action="ignore")  # , category=RuntimeWarning)

# set the start method for multiprocessing
torch.multiprocessing.set_start_method("spawn", force=True)

# set some environment variables
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["OC_CAUSE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

USER = "qsimeon"  # OpenMind username

NUM_NEURONS = 302  # number of neurons in the model organism

MAX_TOKEN_LEN = 1000  # maximum attention block size for Transformer models

RAW_DATA_URL = "https://www.dropbox.com/s/45yqpvtsncx4095/raw_data.zip?dl=1"

# essential raw data files that must be in the raw data directory
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
if torch.backends.mps.is_available():
    print("MPS device found.")
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    print("CUDA device found.")
    DEVICE = torch.device("cuda")
else:
    print("Defaulting to CPU.")
    DEVICE = torch.device("cpu")

# set real C. elegans datasets we have processed
VALID_DATASETS = {
    "Uzel2022",
    "Skora2018",
    "Nichols2017",
    "Kato2015",
    "Kaplan2020",
    "Yemini2021",
    "Leifer2023",  # Different type of dataset: stimulus-response.
    "Flavell2023",  # TODO: something wrong with worm0.
}

SYNTHETIC_DATASETS = {
    "Sines0000",  # Dataset created with the `CreateSyntheticDataset.ipynb` notebook.
    "Lorenz0000",  # Dataset created with the `CreateSyntheticDataset.ipynb` notebook.
    # "Custom<xxxx>",  # Dataset created when `save_datasets` is True in the dataset.yaml config.
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
    Initialize random seeds for random, numpy, torch and cuda.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    return None
