import torch
import random
import os
import re
import math
import hydra
import mat73
import pickle
import h5py
import shutil
import subprocess
import numpy as np
import pandas as pd
from scipy.io import loadmat
from omegaconf import DictConfig, OmegaConf
from utils import ROOT_DIR, RAW_FILES, NUM_NEURONS, NEURONS_302, VALID_DATASETS
from sklearn import preprocessing
from pysindy.differentiation import SmoothedFiniteDifference
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.utils import coalesce
from torch_geometric.data import Data, download_url, extract_zip
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from scipy.linalg import solve
from typing import Tuple, Union, Callable, Dict
from torch.autograd import Variable
