import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCN
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import coalesce
import torch_geometric.transforms as T
import numpy as np
import pandas as pd
import torch_geometric.utils

#Load the c. elegans connectome graph 
from ..data.data_loader import CElegansGraph

dataset = CElegansGraph()
graph = dataset[0]

#@title Investigate the C. elegans graph.
print("Attributes:", "\n", graph.keys, "\n",
      f"Num. nodes {graph.num_nodes}, Num. edges {graph.num_edges}, " 
      f"Num. node features {graph.num_node_features}", end="\n")
print(f"\tHas isolated nodes: {graph.has_isolated_nodes()}")
print(f"\tHas self-loops: {graph.has_self_loops()}")
print(f"\tIs undirected: {graph.is_undirected()}")
print(f"\tIs directed: {graph.is_directed()}") 


#@title Draw the graph of the C. elegans connectome
G = torch_geometric.utils.to_networkx(graph)
pos = nx.kamada_kawai_layout(G) # BEST
# pos = nx.random_layout(G) 
# pos = nx.shell_layout(G) 
# pos = nx.spectral_layout(G) 
# pos = nx.spiral_layout(G) 
# pos = nx.spring_layout(G) 

options = {"edgecolors": "tab:gray", "node_size": 400, "alpha": 0.7}
sensory = [node for i,node in enumerate(G.nodes) if graph.y[i]==0.]
inter = [node for i,node in enumerate(G.nodes) if graph.y[i]==1.]
motor = [node for i,node in enumerate(G.nodes) if graph.y[i]==2.]

nx.draw_networkx_nodes(G, pos, nodelist=sensory, node_color="tab:blue", **options);
nx.draw_networkx_nodes(G, pos, nodelist=inter, node_color="tab:red", **options);
nx.draw_networkx_nodes(G, pos, nodelist=motor, node_color="tab:green", **options);

labels = list(G.nodes)
nx.draw_networkx_labels(G, pos, labels, font_size=8);

junctions = [edge for i,edge in enumerate(G.edges) if graph.edge_attr[i,0]==0.]
synapses = [edge for i,edge in enumerate(G.edges) if graph.edge_attr[i,1]==0.]

nx.draw_networkx_edges(G, pos, edgelist=junctions, alpha=0.5, edge_color="tab:blue");
nx.draw_networkx_edges(G, pos, edgelist=synapses, alpha=0.5, edge_color="tab:red");


# We have a neuron atlas in the file LowResAtlasWithHighResHeadsAndTails.csv with neuron names mapped to 3D coordinates. 
# Load this data and use the first two coordinates (2D) from it for the pos argument to nx.draw_networkx.

#@title Draw using atlas coordinates.

neuron_names = [v.replace('0','') if not v.endswith('0') else v for v in id_neuron.values()]
df = pd.read_csv('LowResAtlasWithHighResHeadsAndTails.csv', header=None, names=['neuron', 'x', 'y', 'z'])
df.head()

assert len(neuron_names) < len(df.neuron)
assert set(neuron_names).issubset(set(df.neuron.values))

keys = labels
values = list(df[df.neuron.isin(neuron_names)][['y', 'z']].values)
my_pos = dict(zip(keys, values))

nx.draw_networkx_nodes(G, my_pos, nodelist=sensory, node_color="tab:blue", **options);
nx.draw_networkx_nodes(G, my_pos, nodelist=inter, node_color="tab:red", **options);
nx.draw_networkx_nodes(G, my_pos, nodelist=motor, node_color="tab:green", **options);
nx.draw_networkx_labels(G, my_pos, labels, font_size=8);
nx.draw_networkx_edges(G, my_pos, edgelist=junctions, alpha=0.5, edge_color="tab:blue");
nx.draw_networkx_edges(G, my_pos, edgelist=synapses, alpha=0.5, edge_color="tab:red");