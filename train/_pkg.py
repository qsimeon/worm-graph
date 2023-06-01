# All imports for train module
import torch
import os
import hydra
import random
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import solve
from typing import Tuple, Union
from datetime import datetime
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from omegaconf import OmegaConf
from utils import DEVICE, LOGS_DIR, NEURONS_302
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
from torch.utils.data import ConcatDataset, DataLoader
from models._utils import NetworkLSTM
from data._utils import NeuralActivityDataset, pick_worm
from data._main import get_dataset
from models._main import get_model
from scipy.signal import savgol_filter