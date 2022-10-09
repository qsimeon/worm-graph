import os; os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys; sys.path.append("..")
import pandas as pd
import networkx as nx
from torch_geometric.utils import to_networkx
from data.data_loader import CElegansDataset

# Load the c. elegans connectome graph 
dataset = CElegansDataset()
graph = dataset[0]

# Draw the graph of the C. elegans connectome.
G = to_networkx(graph)
pos = nx.kamada_kawai_layout(G) # BEST

options = {"edgecolors": "tab:gray", "node_size": 400, "alpha": 0.7}
sensory = [node for i,node in enumerate(G.nodes) if graph.y[i]==0.]
inter = [node for i,node in enumerate(G.nodes) if graph.y[i]==1.]
motor = [node for i,node in enumerate(G.nodes) if graph.y[i]==2.]

nx.draw_networkx_nodes(G, pos, nodelist=sensory, node_color="tab:blue", **options);
nx.draw_networkx_nodes(G, pos, nodelist=inter, node_color="tab:red", **options);
nx.draw_networkx_nodes(G, pos, nodelist=motor, node_color="tab:green", **options);

labels = graph.id_nodes
nx.draw_networkx_labels(G, pos, labels, font_size=8);

junctions = [edge for i,edge in enumerate(G.edges) if graph.edge_attr[i,0]==0.]
synapses = [edge for i,edge in enumerate(G.edges) if graph.edge_attr[i,1]==0.]

nx.draw_networkx_edges(G, pos, edgelist=junctions, alpha=0.5, edge_color="tab:blue");
nx.draw_networkx_edges(G, pos, edgelist=synapses, alpha=0.5, edge_color="tab:red");

# Redraw the graph using atlas coordinates.
pos = graph.pos

nx.draw_networkx_nodes(G, pos, nodelist=sensory, node_color="tab:blue", **options);
nx.draw_networkx_nodes(G, pos, nodelist=inter, node_color="tab:red", **options);
nx.draw_networkx_nodes(G, pos, nodelist=motor, node_color="tab:green", **options);

nx.draw_networkx_labels(G, pos, labels, font_size=8);

nx.draw_networkx_edges(G, pos, edgelist=junctions, alpha=0.5, edge_color="tab:blue");
nx.draw_networkx_edges(G, pos, edgelist=synapses, alpha=0.5, edge_color="tab:red");

if __name__ == "__main__":
      # Investigate the C. elegans graph.
      print("Hello")
      print("Attributes:", "\n", graph.keys, "\n",
            f"Num. nodes {graph.num_nodes}, Num. edges {graph.num_edges}, " 
            f"Num. node features {graph.num_node_features}", end="\n")
      print(f"\tHas isolated nodes: {graph.has_isolated_nodes()}")
      print(f"\tHas self-loops: {graph.has_self_loops()}")
      print(f"\tIs undirected: {graph.is_undirected()}")
      print(f"\tIs directed: {graph.is_directed()}") 