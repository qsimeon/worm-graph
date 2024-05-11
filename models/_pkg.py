# All imports for models module
import os
import time
import math
import torch
import logging

from ncps.torch import CfC
from scipy.stats import norm
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Union
from prettytable import PrettyTable
from scipy.signal import fftconvolve
from torch_geometric.data import Data
from omegaconf import DictConfig, OmegaConf
from einops import rearrange, repeat, einsum
from torch_geometric.utils import to_dense_adj
from torch.autograd.gradcheck import gradcheck
from utils import DEVICE, ROOT_DIR, BLOCK_SIZE, NUM_TOKENS, NUM_NEURONS, VERSION_2
