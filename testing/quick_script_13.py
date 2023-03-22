import torch
from torch_geometric.data import Data
import os

from utils import ROOT_DIR

from data._main import *

# load the raw data
graph_tensors = torch.load(os.path.join(ROOT_DIR, "data/processed/connectome", "graph_tensors.pt"))

if __name__ == "__main__":
    # make the graph
    graph = Data(**graph_tensors)

    dataset = load_Uzel2022()
    print(dataset["worm0"].keys())

    slot = dataset["worm0"]["slot_to_neuron"]
    print(graph.y)
    print(slot)


