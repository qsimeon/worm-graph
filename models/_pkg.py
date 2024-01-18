# All imports for models module
import os
import time
import math
import torch
import logging

from ncps.torch import CfC
from scipy.stats import norm
from typing import Callable, Union
from prettytable import PrettyTable
from omegaconf import DictConfig, OmegaConf
from utils import DEVICE, ROOT_DIR, BLOCK_SIZE, NUM_TOKENS, VERSION_2, VQ_VAE
