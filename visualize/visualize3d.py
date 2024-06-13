import pandas as pd
import os
import torch
import numpy as np

from utils import ROOT_DIR
from torch_geometric.data import Data
from visualize._utils import draw_connectome

from sklearn import preprocessing

import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib import animation


df = pd.read_csv("./data/raw/OpenWormConnectome.csv")
        
origin = []
target = []
edges = []

edge_attr = []

# CANL, CANR not considered neurons in Cook et al (still include in matrix - data will be zero/null when passed to model)

for i in range(len(df)):
    neuron1 = df.loc[i, "Origin"]
    neuron2 = df.loc[i, "Target"]
    
    origin += [neuron1]
    target += [neuron2]
    
    type = df.loc[i, "Type"]
    num_connections = df.loc[i, "Number of Connections"]
    
    if [neuron1, neuron2] not in edges:
        edges += [[neuron1, neuron2]]
        if type == "GapJunction":
            edge_attr += [[num_connections, 0]]
        else:
            edge_attr += [[0, num_connections]]
    else:
        if type == "GapJunction":
            edge_attr[-1][0] = num_connections
        else:
            edge_attr[-1][-1] = num_connections

NEURONS_302 = [ # TODO: Cite source of this list.
            "ADAL", "ADAR", "ADEL", "ADER", "ADFL", "ADFR", "ADLL", "ADLR", "AFDL", "AFDR",
            "AIAL", "AIAR", "AIBL", "AIBR", "AIML", "AIMR", "AINL", "AINR", "AIYL", "AIYR",
            "AIZL", "AIZR", "ALA", "ALML", "ALMR", "ALNL", "ALNR", "AQR", "AS1", "AS10",
            "AS11", "AS2", "AS3", "AS4", "AS5", "AS6", "AS7", "AS8", "AS9", "ASEL", "ASER",
            "ASGL", "ASGR", "ASHL", "ASHR", "ASIL", "ASIR", "ASJL", "ASJR", "ASKL", "ASKR",
            "AUAL", "AUAR", "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER",
            "AVFL", "AVFR", "AVG", "AVHL", "AVHR", "AVJL", "AVJR", "AVKL", "AVKR", "AVL",
            "AVM", "AWAL", "AWAR", "AWBL", "AWBR", "AWCL", "AWCR", "BAGL", "BAGR", "BDUL",
            "BDUR", "CANL", "CANR", "CEPDL", "CEPDR", "CEPVL", "CEPVR", "DA1", "DA2", "DA3",
            "DA4", "DA5", "DA6", "DA7", "DA8", "DA9", "DB1", "DB2", "DB3", "DB4", "DB5",
            "DB6", "DB7", "DD1", "DD2", "DD3", "DD4", "DD5", "DD6", "DVA", "DVB", "DVC",
            "FLPL", "FLPR", "HSNL", "HSNR", "I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5",
            "I6", "IL1DL", "IL1DR", "IL1L", "IL1R", "IL1VL", "IL1VR", "IL2DL", "IL2DR", "IL2L",
            "IL2R", "IL2VL", "IL2VR", "LUAL", "LUAR", "M1", "M2L", "M2R", "M3L", "M3R", "M4",
            "M5", "MCL", "MCR", "MI", "NSML", "NSMR", "OLLL", "OLLR", "OLQDL", "OLQDR",
            "OLQVL", "OLQVR", "PDA", "PDB", "PDEL", "PDER", "PHAL", "PHAR", "PHBL", "PHBR",
            "PHCL", "PHCR", "PLML", "PLMR", "PLNL", "PLNR", "PQR", "PVCL", "PVCR", "PVDL",
            "PVDR", "PVM", "PVNL", "PVNR", "PVPL", "PVPR", "PVQL", "PVQR", "PVR", "PVT",
            "PVWL", "PVWR", "RIAL", "RIAR", "RIBL", "RIBR", "RICL", "RICR", "RID", "RIFL",
            "RIFR", "RIGL", "RIGR", "RIH", "RIML", "RIMR", "RIPL", "RIPR", "RIR", "RIS",
            "RIVL", "RIVR", "RMDDL", "RMDDR", "RMDL", "RMDR", "RMDVL", "RMDVR", "RMED",
            "RMEL", "RMER", "RMEV", "RMFL", "RMFR", "RMGL", "RMGR", "RMHL", "RMHR", "SAADL",
            "SAADR", "SAAVL", "SAAVR", "SABD", "SABVL", "SABVR", "SDQL", "SDQR", "SIADL",
            "SIADR", "SIAVL", "SIAVR", "SIBDL", "SIBDR", "SIBVL", "SIBVR", "SMBDL", "SMBDR",
            "SMBVL", "SMBVR", "SMDDL", "SMDDR", "SMDVL", "SMDVR", "URADL", "URADR", "URAVL",
            "URAVR", "URBL", "URBR", "URXL", "URXR", "URYDL", "URYDR", "URYVL", "URYVR",
            "VA1", "VA10", "VA11", "VA12", "VA2", "VA3", "VA4", "VA5", "VA6", "VA7", "VA8",
            "VA9", "VB1", "VB10", "VB11", "VB2", "VB3", "VB4", "VB5", "VB6", "VB7", "VB8",
            "VB9", "VC1", "VC2", "VC3", "VC4", "VC5", "VC6", "VD1", "VD10", "VD11", "VD12",
            "VD13", "VD2", "VD3", "VD4", "VD5", "VD6", "VD7", "VD8", "VD9"
        ]

neuron_to_idx = dict(zip(NEURONS_302, [i for i in range(len(NEURONS_302))]))
idx_to_neuron = dict(zip([i for i in range(len(NEURONS_302))], NEURONS_302))

edge_index = torch.tensor([[neuron_to_idx[neuron1], neuron_to_idx[neuron2]] for neuron1, neuron2 in edges]).T
node_type = {0: 'Type1', 1: 'Type2'}
num_classes = len(node_type)
n_id = torch.tensor([i for i in range(len(NEURONS_302))])

# for x, y values
# Neurons involved in chemical synapses
GHermChem_Nodes = pd.read_csv("./data/raw/GHermChem_Nodes.csv")  # nodes
neurons_all = set(NEURONS_302)

df = GHermChem_Nodes
df["Name"] = [v.replace("0", "") if not v.endswith("0") else v for v in df["Name"]]
Gsyn_nodes = df[df["Name"].isin(neurons_all)].sort_values(by=["Name"]).reset_index()

le = preprocessing.LabelEncoder()
le.fit(Gsyn_nodes.Group.values)
# num_classes = len(le.classes_)
y = torch.tensor(le.transform(Gsyn_nodes.Group.values), dtype=torch.int32)
x = torch.randn(len(NEURONS_302), 1024, dtype=torch.float)

temp_graph_tensors = torch.load(
    os.path.join(ROOT_DIR, "data", "processed", "connectome", "graph_tensors.pt")
)

# pos = dict(zip([i for i in range(len(NEURONS_302))], [np.random.randn(2) for i in range(len(NEURONS_302))]))
pos = temp_graph_tensors["pos"]
edge_attr = torch.tensor(edge_attr)

graph_tensors = {
    "edge_index": edge_index,
    "edge_attr": edge_attr,
    "pos": pos,
    "num_classes": num_classes,
    "x": x,
    "y": y,
    "idx_to_neuron": idx_to_neuron,
    "node_type": node_type,
    "n_id": n_id,
}

# make the graph
graph = Data(**graph_tensors)

# investigate the graph
print(
    "Attributes:",
    "\n",
    graph.keys,
    "\n",
    f"Num. nodes {graph.num_nodes}, Num. edges {graph.num_edges}, "
    f"Num. node features {graph.num_node_features}",
    end="\n",
)
print(f"\tHas isolated nodes: {graph.has_isolated_nodes()}")
print(f"\tHas self-loops: {graph.has_self_loops()}")
print(f"\tIs undirected: {graph.is_undirected()}")
print(f"\tIs directed: {graph.is_directed()}")

# draw the connectome
# draw_connectome(graph)

print(graph.edge_index.shape)
print(graph.edge_attr.shape)
print(graph.pos)

# processing nodes w/ temporary dummy z value
nodes = np.array([list(node) + [node[1]] for node in graph.pos.values()])
edges = []

for edge in graph.edge_index.T:
    edges += [[nodes[edge[0]], nodes[edge[1]]]]

edges = np.array(edges)

print(edges.shape)
print(nodes.shape)

def init():
    ax.scatter(*nodes.T, alpha=0.2, s=100, color="blue")
    final = []
    for vizedge in edges:
        final = ax.plot(*vizedge.T, color="gray")
    ax.grid(False)
    ax.set_axis_off()
    plt.tight_layout()
    
    return final


def _frame_update(index):
    ax.view_init(index * 0.2, index * 0.5)
    return


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ani = animation.FuncAnimation(
    fig,
    _frame_update,
    init_func=init,
    frames=200,
    interval=20,
    blit=True
)

ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()