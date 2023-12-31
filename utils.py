import os
import torch
import random
import warnings
import numpy as np
import pandas as pd
import torch.multiprocessing

# Ignore all warnings
warnings.filterwarnings(action="ignore")  # , category=RuntimeWarning)

# Set the start method for multiprocessing
torch.multiprocessing.set_start_method("spawn", force=True)

# Set some environment variables
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["OC_CAUSE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

USER = "qsimeon"  # OpenMind username

NUM_NEURONS = 302  # number of neurons in the model organism

BLOCK_SIZE = 5000  # maximum attention block size for Transformer models

NUM_TOKENS = 16384  # number of tokens in the neural vocabulary

RAW_DATA_URL = "https://www.dropbox.com/s/45yqpvtsncx4095/raw_data.zip?dl=1"

RAW_ZIP = "raw_data.zip"

# Essential raw data files that must be in the raw data directory
RAW_FILES = [
    "GHermChem_Edges.csv",
    "GHermChem_Nodes.csv",
    "GHermElec_Sym_Edges.csv",
    "GHermElec_Sym_Nodes.csv",
    "LowResAtlasWithHighResHeadsAndTails.csv",
    "neurons_302.txt",
]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")

LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Get GPU if available
if torch.backends.mps.is_available():
    print("MPS device found.")
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    print("CUDA device found.")
    DEVICE = torch.device("cuda")
else:
    print("Defaulting to CPU.")
    DEVICE = torch.device("cpu")

# Set real C. elegans datasets we have processed
VALID_DATASETS = {
    "Leifer2023",  # Different type of dataset: stimulus-response.
    "Flavell2023",  # TODO: Something wrong with worm0.
    "Uzel2022",
    "Yemini2021",
    "Kaplan2020",
    "Skora2018",
    "Nichols2017",
    "Kato2015",
}

SYNTHETIC_DATASETS = (
    {  # Datasets created with the `CreateSyntheticDataset.ipynb` notebook.
        "Sines0000",
        "Lorenz0000",
        "WhiteNoise0000",
        "RandWalk0000",
        "VanDerPol0000",
        "Shakespeare0000",
    }
)

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
    NEURONS_302 = [
        "ADAL",
        "ADAR",
        "ADEL",
        "ADER",
        "ADFL",
        "ADFR",
        "ADLL",
        "ADLR",
        "AFDL",
        "AFDR",
        "AIAL",
        "AIAR",
        "AIBL",
        "AIBR",
        "AIML",
        "AIMR",
        "AINL",
        "AINR",
        "AIYL",
        "AIYR",
        "AIZL",
        "AIZR",
        "ALA",
        "ALML",
        "ALMR",
        "ALNL",
        "ALNR",
        "AQR",
        "AS1",
        "AS10",
        "AS11",
        "AS2",
        "AS3",
        "AS4",
        "AS5",
        "AS6",
        "AS7",
        "AS8",
        "AS9",
        "ASEL",
        "ASER",
        "ASGL",
        "ASGR",
        "ASHL",
        "ASHR",
        "ASIL",
        "ASIR",
        "ASJL",
        "ASJR",
        "ASKL",
        "ASKR",
        "AUAL",
        "AUAR",
        "AVAL",
        "AVAR",
        "AVBL",
        "AVBR",
        "AVDL",
        "AVDR",
        "AVEL",
        "AVER",
        "AVFL",
        "AVFR",
        "AVG",
        "AVHL",
        "AVHR",
        "AVJL",
        "AVJR",
        "AVKL",
        "AVKR",
        "AVL",
        "AVM",
        "AWAL",
        "AWAR",
        "AWBL",
        "AWBR",
        "AWCL",
        "AWCR",
        "BAGL",
        "BAGR",
        "BDUL",
        "BDUR",
        "CANL",
        "CANR",
        "CEPDL",
        "CEPDR",
        "CEPVL",
        "CEPVR",
        "DA1",
        "DA2",
        "DA3",
        "DA4",
        "DA5",
        "DA6",
        "DA7",
        "DA8",
        "DA9",
        "DB1",
        "DB2",
        "DB3",
        "DB4",
        "DB5",
        "DB6",
        "DB7",
        "DD1",
        "DD2",
        "DD3",
        "DD4",
        "DD5",
        "DD6",
        "DVA",
        "DVB",
        "DVC",
        "FLPL",
        "FLPR",
        "HSNL",
        "HSNR",
        "I1L",
        "I1R",
        "I2L",
        "I2R",
        "I3",
        "I4",
        "I5",
        "I6",
        "IL1DL",
        "IL1DR",
        "IL1L",
        "IL1R",
        "IL1VL",
        "IL1VR",
        "IL2DL",
        "IL2DR",
        "IL2L",
        "IL2R",
        "IL2VL",
        "IL2VR",
        "LUAL",
        "LUAR",
        "M1",
        "M2L",
        "M2R",
        "M3L",
        "M3R",
        "M4",
        "M5",
        "MCL",
        "MCR",
        "MI",
        "NSML",
        "NSMR",
        "OLLL",
        "OLLR",
        "OLQDL",
        "OLQDR",
        "OLQVL",
        "OLQVR",
        "PDA",
        "PDB",
        "PDEL",
        "PDER",
        "PHAL",
        "PHAR",
        "PHBL",
        "PHBR",
        "PHCL",
        "PHCR",
        "PLML",
        "PLMR",
        "PLNL",
        "PLNR",
        "PQR",
        "PVCL",
        "PVCR",
        "PVDL",
        "PVDR",
        "PVM",
        "PVNL",
        "PVNR",
        "PVPL",
        "PVPR",
        "PVQL",
        "PVQR",
        "PVR",
        "PVT",
        "PVWL",
        "PVWR",
        "RIAL",
        "RIAR",
        "RIBL",
        "RIBR",
        "RICL",
        "RICR",
        "RID",
        "RIFL",
        "RIFR",
        "RIGL",
        "RIGR",
        "RIH",
        "RIML",
        "RIMR",
        "RIPL",
        "RIPR",
        "RIR",
        "RIS",
        "RIVL",
        "RIVR",
        "RMDDL",
        "RMDDR",
        "RMDL",
        "RMDR",
        "RMDVL",
        "RMDVR",
        "RMED",
        "RMEL",
        "RMER",
        "RMEV",
        "RMFL",
        "RMFR",
        "RMGL",
        "RMGR",
        "RMHL",
        "RMHR",
        "SAADL",
        "SAADR",
        "SAAVL",
        "SAAVR",
        "SABD",
        "SABVL",
        "SABVR",
        "SDQL",
        "SDQR",
        "SIADL",
        "SIADR",
        "SIAVL",
        "SIAVR",
        "SIBDL",
        "SIBDR",
        "SIBVL",
        "SIBVR",
        "SMBDL",
        "SMBDR",
        "SMBVL",
        "SMBVR",
        "SMDDL",
        "SMDDR",
        "SMDVL",
        "SMDVR",
        "URADL",
        "URADR",
        "URAVL",
        "URAVR",
        "URBL",
        "URBR",
        "URXL",
        "URXR",
        "URYDL",
        "URYDR",
        "URYVL",
        "URYVR",
        "VA1",
        "VA10",
        "VA11",
        "VA12",
        "VA2",
        "VA3",
        "VA4",
        "VA5",
        "VA6",
        "VA7",
        "VA8",
        "VA9",
        "VB1",
        "VB10",
        "VB11",
        "VB2",
        "VB3",
        "VB4",
        "VB5",
        "VB6",
        "VB7",
        "VB8",
        "VB9",
        "VC1",
        "VC2",
        "VC3",
        "VC4",
        "VC5",
        "VC6",
        "VD1",
        "VD10",
        "VD11",
        "VD12",
        "VD13",
        "VD2",
        "VD3",
        "VD4",
        "VD5",
        "VD6",
        "VD7",
        "VD8",
        "VD9",
    ]


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
