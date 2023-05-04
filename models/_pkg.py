# All imports for models module
import torch
import math
import os
import hydra
from typing import Callable, Union
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from utils import DEVICE, ROOT_DIR
from torch_geometric.nn import GCNConv
from omegaconf import DictConfig, OmegaConf
