import os
import torch
from utils import ROOT_DIR, NEURONS_302
from torch_geometric.data import Data
from tests.leandro.plots import *
import base64
from IPython.display import Image, display
import requests
from tests.leandro.hierarchical_clustering_analysis import load_reference

class MiniGraph:
    def __init__(self, direction='TD', group_by='four'):
        self.direction = direction.upper()
        self.minigraph = f"""graph {direction};\n"""
        self.nodes = []
        self.ref_dict = load_reference(group_by=group_by)
        self.color_dict = {
            'I': '#FF1F5B',
            'M': '#00CD6C',
            'S': '#009ADE',
            'P': '#AF58BA',
            'MI': '#FFC61E',
            'SM': '#F28522',
            'SI': '#A0B1BA',
            'SMI': '#A6761D',
            'U': '#E9002D'
        }

    def restart(self):
        self.minigraph = f"""graph {self.direction};\n"""
        self.nodes = []

    def add_nodes(self, neuron, connected_neurons):
        assert isinstance(neuron, str), 'Neuron must be a string'
        assert isinstance(connected_neurons, list), 'Connected_neurons must be a list'
        # Add lines to the graph with the specified color
        self.minigraph += '    {} --> {};\n'.format(neuron, ' & '.join(connected_neurons))
        # Add the neuron to the list of nodes
        self.nodes.append(neuron)
        # Add the connected neurons to the list of nodes
        self.nodes.extend(connected_neurons)
        # Remove duplicates
        self.nodes = list(set(self.nodes))

    def add_color(self):
        for neuron_class, color in self.color_dict.items():
            self.minigraph += f'    classDef {neuron_class} fill: {color}, stroke:#000000, stroke-width:1px;\n'

        for neuron_name in self.nodes:
            self.minigraph += f'    class {neuron_name} {self.ref_dict[neuron_name]};\n'

    def display(self, save=False, filename='graph.png'):
        self.add_color()
        graphbytes = self.minigraph.encode("ascii")
        base64_bytes = base64.b64encode(graphbytes)
        base64_string = base64_bytes.decode("ascii")
        image_url = "https://mermaid.ink/img/" + base64_string
        display(Image(url=image_url))

        if save:
            # Save the image from the URL
            response = requests.get(image_url)
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Graph saved as {filename}")

    def print(self):
        print(self.minigraph)


class MiniConnectome():

    def __init__(self, direction='TD', group_by='four'):
        super().__init__()
        self.load_connectome()
        self.minigraph = MiniGraph(direction, group_by)

    def load_connectome(self):
        # load the connectome data
        graph_tensors = torch.load(
            os.path.join(ROOT_DIR, "data", "processed", "connectome", "graph_tensors.pt")
        )

        # make the graph
        self.graph = Data(**graph_tensors)
        # print(self.graph.edge_index.shape) # Row 0: source neuron, Row 1: target neuron
        # print(self.graph.edge_attr.shape) # Column 0: number of gap junction connections, Column 1: number of chemical synapse connections

    def get_connected_neurons(self, neuron_name, index=False):

        neuron_to_idx = {u: v for (v, u) in self.graph.idx_to_neuron.items()}

        # Get the edge indices corresponding to the given node
        node_index = neuron_to_idx[neuron_name]
        connected_edge_indices = (self.graph.edge_index[0] == node_index).nonzero()

        # Get the connected nodes by extracting the second row of the connected_edge_indices
        connected_nodes = self.graph.edge_index[1, connected_edge_indices].flatten()
        
        if index == True:
            return connected_nodes.detach().numpy().tolist()
        else:
            return [self.graph.idx_to_neuron[idx] for idx in connected_nodes.detach().numpy()]
        
    def get_neuron_by_number_of_connections(self, N=1):
        num_conections = []
        for neuron_name in NEURONS_302:
            num_conections.append(self.get_connected_neurons(neuron_name, index=False))
        # Name of all neurons with N connections
        print('Neurons with {} connections: {}'.format(N, [NEURONS_302[i] for i, x in enumerate(num_conections) if len(x) == N]))

        return [NEURONS_302[i] for i, x in enumerate(num_conections) if len(x) == N]