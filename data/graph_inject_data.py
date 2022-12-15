import torch
from torch_geometric.data import Data


def graph_inject_data(single_worm_dataset, connectome_graph):
  """
  Find the nodes on the connecotme corresponding to labelled 
  neurons in the provided single worm dataset and place the data 
  on the connectome graph.
  Returns the full graph with 0s on unlabelled neurons, 
  the subgraph with only labelled neurons, the subgraph mask.
  """
  calcium_data = single_worm_dataset['data']
  graph = connectome_graph
  # get the calcium data for this worm
  dataset = calcium_data.squeeze()
  max_time, num_neurons = dataset.shape
  assert max_time == single_worm_dataset['max_time']
  assert num_neurons == single_worm_dataset['num_neurons']
  print("How much real data do we have?", dataset.shape) # (time, neurons)
  print("Current data on connectome graph:", graph.x.cpu().numpy().shape) # (neurons, time)
  # find the graph nodes matching the neurons in the dataset 
  neuron_ids = single_worm_dataset['neuron_ids']
  graph_inds = [k-1 for k,v in graph.id_neuron.items() if 
                v in set(neuron_ids.values())] # neuron indices in connectome
  data_inds = [k_-1 for k_,v_ in neuron_ids.items() if 
              v_ in set(graph.id_neuron.values())] # neuron indices in sparse dataset
  # 'inject' the data by creating a clone graph with the desired features
  new_x = torch.zeros(graph.num_nodes, max_time, dtype=torch.float64)
  new_x[graph_inds, :] = dataset[:, data_inds].T
  graph = Data(x=new_x, y=graph.y, edge_index=graph.edge_index, edge_attr=graph.edge_attr, 
                  node_type=graph.node_type, pos=graph.pos, id_neuron=graph.id_neuron)
  # assign each node its global node index
  graph.n_id = torch.arange(graph.num_nodes)
  # create the subgraph that has labelled neurons and data
  subgraph_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
  subgraph_mask.index_fill_(0, torch.tensor(graph_inds).long(), 1).bool()
  # extract out the subgraph
  subgraph = graph.subgraph(subgraph_mask)
  # reset neuron indices for labeling
  subgraph.id_neuron = {i: graph.id_neuron[k] for i,k 
                    in enumerate(subgraph.n_id.numpy())}
  subgraph.pos = {i: graph.pos[k] for i,k in enumerate(subgraph.n_id.numpy())}
  # check out the new attributes
  print("Attributes:", "\n", subgraph.keys, "\n",
      f"Num. nodes {subgraph.num_nodes}, Num. edges {subgraph.num_edges}, " 
      f"Num. node features {subgraph.num_node_features}", end="\n")
  print(f"\tHas isolated nodes: {subgraph.has_isolated_nodes()}")
  print(f"\tHas self-loops: {subgraph.has_self_loops()}")
  print(f"\tIs undirected: {subgraph.is_undirected()}")
  print(f"\tIs directed: {subgraph.is_directed()}") 
  # return the graph, subgraph and mask
  return graph, subgraph, subgraph_mask