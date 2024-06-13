from ursina import *
from ursina.shaders import lit_with_shadows_shader 

import random

import numpy as np

import torch


random.seed(0)

num_nodes = 302

rotation_speed_scale = 1

app = Ursina()
app.setBackgroundColor(rgb(0.1, 0.1, 0.1))

graph_tensors = torch.load("../data/processed/connectome/open_worm_graph_tensors.pt")
pos = np.array(list(graph_tensors["pos"].values()))

nodes = np.array([[neuron[0]-np.mean(pos[:, 0]), neuron[1]-np.mean(pos[:, 1]), random.uniform(-560, 560)] for neuron in pos])
nodes[:, 0] /= np.max(nodes[:, 0])
nodes[:, 1] /= np.max(nodes[:, 1])
nodes[:, 2] /= np.max(nodes[:, 2])

nodes *= 10

edges = graph_tensors["edge_index"].T.numpy()
edge_attr = graph_tensors["edge_attr"].numpy()

center = Entity()

class Graph(Entity):
    def __init__(self):
        self.nodes_ = [Entity(model='sphere', scale=(0.2, 0.2, 0.2), color=color.white, shader=lit_with_shadows_shader) for i in range(len(nodes))]
        for i, node in enumerate(self.nodes_):
            node.x, node.y, node.z = nodes[i]
            node.parent = center
        
        self.edges = [Entity(model='cube', color=rgb(0.0, 1, 0.5), shader=lit_with_shadows_shader) for i in range(len(edges))]
        for i, edge in enumerate(self.edges):
            node1 = self.nodes_[edges[i][0]]
            node2 = self.nodes_[edges[i][1]]
            
            edge.x = (node1.x + node2.x) / 2
            edge.y = (node1.y + node2.y) / 2
            edge.z = (node1.z + node2.z) / 2
            
            edge.scale = (0.05/np.max(edge_attr)*(edge_attr[i][0]+edge_attr[i][1]), 0.05/np.max(edge_attr)*(edge_attr[i][0]+edge_attr[i][1]), np.sqrt((node1.x-node2.x)**2+(node1.y-node2.y)**2+(node1.z-node2.z)**2))
            
            edge.parent = center
            edge.look_at(node1)
        
    def update(self):
        child = center
        # rotation
        if held_keys['w']:
            child.rotation_x += rotation_speed_scale
        if held_keys['s']:
            child.rotation_x -= rotation_speed_scale
        if held_keys['d']:
            child.world_rotation_y_setter(child.world_rotation_y_getter() + rotation_speed_scale)
        if held_keys['a']:
            child.world_rotation_y_setter(child.world_rotation_y_getter() - rotation_speed_scale)

        # translation
        if held_keys['up arrow']:
            child.z -= 0.1
        if held_keys['down arrow']:
            child.z += 0.1
        if held_keys['right arrow']:
            child.x -= 0.1
        if held_keys['left arrow']:
            child.x += 0.1
            

graph = Graph()

def update():
    graph.update()

app.run()