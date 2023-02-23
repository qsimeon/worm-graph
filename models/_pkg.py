# All imports for models module
import torch
import hydra
from typing import Callable
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from torch_geometric.nn import GCNConv
from omegaconf import DictConfig, OmegaConf
