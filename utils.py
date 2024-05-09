# NOTE: To oavoid circular imports, only import libraries essential to
# initializing global variables and functions used by the main.py script.
import os
import torch
import random
import logging
import warnings
import numpy as np
import pandas as pd
import torch.multiprocessing

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ignore all warnings
warnings.filterwarnings(action="ignore")

# Set the start method for multiprocessing
torch.multiprocessing.set_start_method("spawn", force=True)

# Set some environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"  # just a random port
os.environ["OC_CAUSE"] = "1"

# Some global variables
USER = "qsimeon"  # OpenMind/computing cluster username

NUM_NEURONS = 302  # number of neurons in the model organism

BLOCK_SIZE = 512  # maximum attention block size to use for Transformers

VERSION_2 = False  # whether to use version 2 of the models (tokenizes neural data)

NUM_TOKENS = 256  # number of tokens in the neural vocabulary if using version 2

WORLD_SIZE = torch.cuda.device_count()

RAW_DATA_URL = "https://www.dropbox.com/scl/fi/q0dg6grqt5nz28dqbxok4/raw_data.zip?rlkey=q7yea001kxuen9w4sedi930oc&dl=1"

RAW_ZIP = "raw_data.zip"

# Essential raw data files that must be in the raw data directory
RAW_FILES = [  # TODO: Cite sources of these files.
    "GHermChem.mat",
    "GHermChem_Edges.csv",
    "GHermChem_Nodes.csv",
    "GHermElec_Sym.mat",
    "GHermElec_Sym_Edges.csv",
    "GHermElec_Sym_Nodes.csv",
    "LowResAtlasWithHighResHeadsAndTails.csv",
    "neurons_302.txt",
]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")

LOGS_DIR = os.path.join(ROOT_DIR, "logs")


# Method for initializing the global computing device
def init_device():
    """
    Initialize the global computing device to be used.
    """
    if torch.backends.mps.is_available():
        print("MPS device found.")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("CUDA device found.")
        device = torch.device("cuda")
        gpu_props = torch.cuda.get_device_properties(device)
        print(f"\t GPU: {gpu_props.name}")
    else:
        print("Defaulting to CPU.")
        device = torch.device("cpu")
    return device


# Get GPU if available
DEVICE = init_device()

# Set real C. elegans datasets we have processed
EXPERIMENT_DATASETS = {
    "Leifer2023",  # Different type of dataset: stimulus-response.
    "Lin2023",
    "Flavell2023",  # TODO: Something is wrong with worm0 always in this dataset. Why?
    "Uzel2022",
    "Yemini2021",
    "Kaplan2020",
    "Skora2018",
    "Nichols2017",
    "Kato2015",
}

SYNTHETIC_DATASETS = {  # Datasets created with the `CreateSyntheticDataset.ipynb` notebook.
    "Sines0000",
    "Lorenz0000",
    "WhiteNoise0000",
    "RandWalk0000",
    "VanDerPol0000",
    "Wikitext0000",
    "Recurrent0000",
}

# List of all 302 hermaphrodite neurons
if os.path.exists(RAW_DATA_DIR):
    NEURONS_302 = sorted(
        pd.read_csv(
            os.path.join(RAW_DATA_DIR, "neurons_302.txt"),
            sep=" ",
            header=None,
            names=["neuron"],
        ).neuron
    )
else:
    # fmt: off
    NEURONS_302 = [ 
            # References: (1) https://www.wormatlas.org/neurons/Individual%20Neurons/Neuronframeset.html 
            #             (2) https://www.wormatlas.org/NeuronNames.htm
            "ADAL", "ADAR", "ADEL", "ADER", "ADFL", "ADFR", "ADLL", "ADLR", "AFDL", "AFDR",
            "AIAL", "AIAR", "AIBL", "AIBR", "AIML", "AIMR", "AINL", "AINR", "AIYL", "AIYR",
            "AIZL", "AIZR", "ALA", "ALML", "ALMR", "ALNL", "ALNR", "AQR", "AS1", "AS10",
            "AS11", "AS2", "AS3", "AS4", "AS5", "AS6", "AS7", "AS8", "AS9", "ASEL", "ASER",
            "ASGL", "ASGR", "ASHL", "ASHR", "ASIL", "ASIR", "ASJL", "ASJR", "ASKL", "ASKR",
            "AUAL", "AUAR", "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER",
            "AVFL", "AVFR", "AVG", "AVHL", "AVHR", "AVJL", "AVJR", "AVKL", "AVKR", "AVL",
            "AVM", "AWAL", "AWAR", "AWBL", "AWBR", "AWCL", "AWCR", "BAGL", "BAGR", "BDUL",
            "BDUR", "CANL", "CANR", "CEPDL", "CEPDR", "CEPVL", "CEPVR", "DA1", "DA2", "DA3",
            "DA4", "DA5", "DA6", "DA7", "DA8", "DA9", "DB1", "DB2", "DB3", "DB4", "DB5",
            "DB6", "DB7", "DD1", "DD2", "DD3", "DD4", "DD5", "DD6", "DVA", "DVB", "DVC",
            "FLPL", "FLPR", "HSNL", "HSNR", "I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5",
            "I6", "IL1DL", "IL1DR", "IL1L", "IL1R", "IL1VL", "IL1VR", "IL2DL", "IL2DR", "IL2L",
            "IL2R", "IL2VL", "IL2VR", "LUAL", "LUAR", "M1", "M2L", "M2R", "M3L", "M3R", "M4",
            "M5", "MCL", "MCR", "MI", "NSML", "NSMR", "OLLL", "OLLR", "OLQDL", "OLQDR",
            "OLQVL", "OLQVR", "PDA", "PDB", "PDEL", "PDER", "PHAL", "PHAR", "PHBL", "PHBR",
            "PHCL", "PHCR", "PLML", "PLMR", "PLNL", "PLNR", "PQR", "PVCL", "PVCR", "PVDL",
            "PVDR", "PVM", "PVNL", "PVNR", "PVPL", "PVPR", "PVQL", "PVQR", "PVR", "PVT",
            "PVWL", "PVWR", "RIAL", "RIAR", "RIBL", "RIBR", "RICL", "RICR", "RID", "RIFL",
            "RIFR", "RIGL", "RIGR", "RIH", "RIML", "RIMR", "RIPL", "RIPR", "RIR", "RIS",
            "RIVL", "RIVR", "RMDDL", "RMDDR", "RMDL", "RMDR", "RMDVL", "RMDVR", "RMED",
            "RMEL", "RMER", "RMEV", "RMFL", "RMFR", "RMGL", "RMGR", "RMHL", "RMHR", "SAADL",
            "SAADR", "SAAVL", "SAAVR", "SABD", "SABVL", "SABVR", "SDQL", "SDQR", "SIADL",
            "SIADR", "SIAVL", "SIAVR", "SIBDL", "SIBDR", "SIBVL", "SIBVR", "SMBDL", "SMBDR",
            "SMBVL", "SMBVR", "SMDDL", "SMDDR", "SMDVL", "SMDVR", "URADL", "URADR", "URAVL",
            "URAVR", "URBL", "URBR", "URXL", "URXR", "URYDL", "URYDR", "URYVL", "URYVR",
            "VA1", "VA10", "VA11", "VA12", "VA2", "VA3", "VA4", "VA5", "VA6", "VA7", "VA8",
            "VA9", "VB1", "VB10", "VB11", "VB2", "VB3", "VB4", "VB5", "VB6", "VB7", "VB8",
            "VB9", "VC1", "VC2", "VC3", "VC4", "VC5", "VC6", "VD1", "VD10", "VD11", "VD12",
            "VD13", "VD2", "VD3", "VD4", "VD5", "VD6", "VD7", "VD8", "VD9"
        ]
    # fmt: on


# Method for globally setting all random seeds
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
