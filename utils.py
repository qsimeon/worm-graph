import os
import torch

# defines `worm_graph` as the root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# get GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set of C. elegans datasets we have processed
VALID_DATASETS = {'Nichols2017', 'Nguyen2017', 'Skora2018','Kaplan2020', 'Uzel2022', 'Kato2015'}