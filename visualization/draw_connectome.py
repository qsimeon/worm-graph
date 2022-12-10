import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch_geometric

def draw_connectome(network, pos=None, labels=None):
  '''
  Args: 
    network: PyG Data object containing a C. elegans connectome graph.
    pos: dict, mapping of node index to 2D coordinate.
    labels: dict, mapping of node index to neuron name.
  '''
  # convert to networkx
  G = torch_geometric.utils.to_networkx(network)

  #create figure
  plt.figure(figsize=(20,10)) 

  ## nodes
  inter = [node for i,node in enumerate(G.nodes) if network.y[i]==0.]
  motor = [node for i,node in enumerate(G.nodes) if network.y[i]==1.]
  other = [node for i,node in enumerate(G.nodes) if network.y[i]==2.]
  pharynx = [node for i,node in enumerate(G.nodes) if network.y[i]==3.]
  sensory = [node for i,node in enumerate(G.nodes) if network.y[i]==4.]
  sexspec = [node for i,node in enumerate(G.nodes) if network.y[i]==5.]

  ## edges
  junctions = [edge for i,edge in enumerate(G.edges) if 
               network.edge_attr[i,0]>0.] # gap junctions/electrical synapses encoded as [1,0]
  synapses = [edge for i,edge in enumerate(G.edges) if 
              network.edge_attr[i,1]>0.] # chemical synapse encoded as [0,1]

  ## edge weights
  gap_weights = [int(network.edge_attr[i,0])/50 for i,edge in enumerate(G.edges)]
  chem_weights = [int(network.edge_attr[i,1])/50 for i,edge in enumerate(G.edges)]

  ## metadata
  if pos is None: pos = network.pos
  # pos = nx.kamada_kawai_layout(G)
  if labels is None: labels = network.id_neuron
  options = {"edgecolors": "tab:gray", "node_size": 500, "alpha": 0.5}

  ## draw nodes
  nx.draw_networkx_edges(G, pos, edgelist=junctions, width=gap_weights, 
                         alpha=0.5, edge_color="tab:blue");
  nx.draw_networkx_edges(G, pos, edgelist=synapses, width=chem_weights,
                         alpha=0.5, edge_color="tab:red");
  nx.draw_networkx_labels(G, pos, labels, font_size=6);

  ## draw edges
  nx.draw_networkx_nodes(G, pos, nodelist=inter, node_color="blue", **options);
  nx.draw_networkx_nodes(G, pos, nodelist=motor, node_color="red", **options);
  nx.draw_networkx_nodes(G, pos, nodelist=other, node_color="green", **options);
  nx.draw_networkx_nodes(G, pos, nodelist=pharynx, node_color="yellow", **options);
  nx.draw_networkx_nodes(G, pos, nodelist=sensory, node_color="magenta", **options);
  nx.draw_networkx_nodes(G, pos, nodelist=sexspec, node_color="cyan", **options);

  legend_elements = [
      Line2D([0], [0], marker='o', color='w', label='inter', markerfacecolor='b', alpha=0.5, markersize=15),
      Line2D([0], [0], marker='o', color='w', label='motor', markerfacecolor='r', alpha=0.5, markersize=15),
      Line2D([0], [0], marker='o', color='w', label='other', markerfacecolor='g', alpha=0.5, markersize=15),
      Line2D([0], [0], marker='o', color='w', label='pharynx', markerfacecolor='y', alpha=0.5, markersize=15),
      Line2D([0], [0], marker='o', color='w', label='sensory', markerfacecolor='m', alpha=0.5, markersize=15),
      Line2D([0], [0], marker='o', color='w', label='sex', markerfacecolor='c', alpha=0.5, markersize=15),
      Line2D([0], [0], color='b', label='gap junction', linewidth=2, alpha=0.5, markersize=10),
      Line2D([0], [0], color='r', label='synapse', linewidth=2, alpha=0.5, markersize=10),        
  ]
  plt.legend(handles=legend_elements, loc='upper right')
  plt.show()