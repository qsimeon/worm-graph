import sys; sys.path.append("..")
from data.data_loader import CElegansDataset

# Load the c. elegans connectome graph 
dataset = CElegansDataset(root="../data/")
graph = dataset[0]

if __name__ == "__main__":
      # Investigate the C. elegans graph
      print()
      print("C. elegans connectome graph loaded successfully!")
      print("Attributes:", "\n", graph.keys, "\n",
            f"Num. nodes {graph.num_nodes}, Num. edges {graph.num_edges}, " 
            f"Num. node features {graph.num_node_features}", end="\n")
      print(f"\tHas isolated nodes: {graph.has_isolated_nodes()}")
      print(f"\tHas self-loops: {graph.has_self_loops()}")
      print(f"\tIs undirected: {graph.is_undirected()}")
      print(f"\tIs directed: {graph.is_directed()}") 
      print()