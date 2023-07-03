# All imports for models module
import torch
import math
import os
import hydra
from torch.cuda.amp import autocast
from typing import Callable, Union
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from utils import DEVICE, ROOT_DIR, MAX_TOKEN_LEN
from torch_geometric.nn import GCNConv
from omegaconf import DictConfig, OmegaConf
