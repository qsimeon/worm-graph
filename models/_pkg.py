# All imports for models module
import copy
import hydra
import math
import os
import torch

from omegaconf import DictConfig, OmegaConf
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from torch.cuda.amp import autocast
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from typing import Callable, Union
from utils import DEVICE, ROOT_DIR, MAX_TOKEN_LEN
