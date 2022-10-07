from preprocessing import WormNeuralDynamicsDataLoader
from preprocessing import *
from torch_geometric_temporal.signal import temporal_signal_split
from train import train
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN


# Load data
loader = WormNeuralDynamicsDataLoader.WormNeuralDynamicsDataLoader()
dataset = loader.get_dataset()
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

# Create network
model = RecurrentGCN(node_features=3)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model = train(train_dataset, model)

# Evaluation
model.eval()
cost = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
cost = cost / (time + 1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))


# Visualize