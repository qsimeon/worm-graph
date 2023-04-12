import torch
from torch_geometric.data import Data
import os

from utils import ROOT_DIR

from data._main import *

# load the raw data
graph_tensors = torch.load(
    os.path.join(ROOT_DIR, "data/processed/connectome", "graph_tensors.pt")
)

if __name__ == "__main__":
    # make the graph
    graph = Data(**graph_tensors)

    dataset = load_Uzel2022()
    print(dataset["worm0"].keys())

    slot = dataset["worm0"]["neuron_to_slot"]
    print(graph.y)
    print(slot)

    list_save = []
    for i in range(302):
        if dataset["worm0"]["named_neurons_mask"][i] == True:
            list_save.append(i)

    print(list_save)

    neuron_range = [12, 22, 59, 96, 216, 228, 260, 274, 275]
    for item in neuron_range:
        print(dataset["worm0"]["slot_to_neuron"][item], graph.y[item].item())
