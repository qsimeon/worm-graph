import os
import torch

# defines `worm_graph` as the root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# get GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')