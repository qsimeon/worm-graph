import logging
import mlflow
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.multiprocessing
import warnings
from omegaconf import DictConfig, ListConfig

# Ignore sklearn's RuntimeWarnings
warnings.filterwarnings(action="ignore", category=RuntimeWarning)

# set the start method for multiprocessing
torch.multiprocessing.set_start_method("spawn", force=True)

# set some environment variables
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["OC_CAUSE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

USER = "qsimeon"  # OpenMind username

NUM_NEURONS = 302  # number of neurons in the model organism

MAX_TOKEN_LEN = 7000  # maximum attention block size for transformer models

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set real C. elegans datasets we have processed
VALID_DATASETS = {
    "Uzel2022",
    "Skora2018",
    "Nichols2017",
    "Kato2015",
    "Kaplan2020",
    "Leifer2023",  # Different type of dataset: stimulus-response.
    "Flavell2023",  # TODO: something wrong with worm0.
}

SYNTHETIC_DATASETS = {"Synthetic0000",
                      "Custom", # Dataset created when 'dataset.save' is True
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

def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)


def string_to_dict(s):
    # Convert stringified dictionary to an actual dictionary
    pairs = s.split(',')
    d = {}
    for pair in pairs:
        k, v = pair.split('=')
        d[k] = int(v)
    return d