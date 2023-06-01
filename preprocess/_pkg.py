# Built-in libraries
import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import subprocess

# Third-party libraries
import derivative
import h5py
import hydra
import mat73
import numpy as np
import pandas as pd
import torch
from derivative import dxdt
from omegaconf import DictConfig, OmegaConf
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.data import Data, download_url, extract_zip
from torch_geometric.utils import coalesce
from typing import Tuple, Union, Callable, Dict

# Local libraries
from utils import ROOT_DIR, RAW_FILES, NUM_NEURONS, NEURONS_302, VALID_DATASETS
from pysindy.differentiation import SmoothedFiniteDifference

from scipy.signal import savgol_filter
from scipy.linalg import solve
from typing import Tuple, Union
from torch.autograd import Variable