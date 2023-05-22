# Built-in libraries
import json
import math
import os
import pickle
import random
import re
import shutil
import subprocess

# Third-party libraries
import h5py
import hydra
import mat73
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.io import loadmat
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.autograd import Variable
from torch_geometric.data import Data, download_url, extract_zip
from torch_geometric.utils import coalesce
from typing import Tuple, Union, Callable, Dict

# Local libraries
from utils import ROOT_DIR, RAW_FILES, NUM_NEURONS, NEURONS_302, VALID_DATASETS
from pysindy.differentiation import SmoothedFiniteDifference
