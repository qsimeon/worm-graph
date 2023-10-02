import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import yaml
import torch
import omegaconf
import logging

from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data
from data._utils import create_combined_dataset, split_combined_dataset
from train._utils import compute_loss_vectorized
from models._main import get_model
from utils import ROOT_DIR, DEVICE

# === Hierarchical clustering ===
import json
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from dtaidistance import dtw
import sklearn.metrics as sm
from visualize._utils import plot_heat_map
from data._main import get_datasets
from matplotlib.patches import Patch
