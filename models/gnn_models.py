import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric_temporal.nn.recurrent import TGCN, DCRNN
from torch_geometric_temporal.nn.recurrent import EvolveGCNH, GConvGRU


class ConvGNN(torch.nn.Module):
    '''TODO: docstrings
    Class to use for subclassing all convoultuional GNNs'''
    def __init__(self):
        super(ConvGNN, self).__init__()


class DeepGCNConv(ConvGNN):
    '''TODO: want classes to be understandable from their names.'''
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(DeepGCNConv, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        batch = graph.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


class DeepGATConv(ConvGNN):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(DeepGATConv, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = DeepGATConv(num_node_features, hidden_channels)
        self.conv2 = DeepGATConv(hidden_channels, hidden_channels)
        self.conv3 = DeepGATConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        batch = graph.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class RecurrentGCN(torch.nn.Module):
    def __init__(self):
        super(RecurrentGCN, self).__init__()


class DeepRGCN(RecurrentGCN):
    '''DCRNN: Deep Convolutional Recurrent Neural Network.'''
    def __init__(self, node_features):
        super(DeepRGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


class EvolveRCGN(RecurrentGCN):
    def __init__(self, node_count, node_features):
        '''EvolveGCNH'''
        super(EvolveRCGN, self).__init__()
        self.recurrent = EvolveGCNH(node_count, node_features)
        self.linear = torch.nn.Linear(node_features, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


class GatedRGCN(RecurrentGCN):
    '''GConvGRU'''
    def __init__(self, node_features):
        super(GatedRGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
        
class GraphNN(torch.nn.Module):
  '''Predict the residual at the next time step, given a history of length L'''
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(GraphNN, self).__init__()
    # need a conv. layer for each type of synapse
    self.elec_conv = GCNConv(in_channels=input_dim, 
                             out_channels=hidden_dim, improved=True)
    self.chem_conv = GCNConv(in_channels=input_dim, 
                             out_channels=hidden_dim, improved=True)
    # readout layer transforms node features to output
    self.linear = torch.nn.Linear(in_features=2*hidden_dim, 
                                  out_features=output_dim)
  
  def forward(self, x, edge_index, edge_attr):
    elec_weight = edge_attr[:,0]
    chem_weight = edge_attr[:,1]
    elec_hid = self.elec_conv(x, edge_index, elec_weight)
    chem_hid = self.chem_conv(x, edge_index, chem_weight)
    hidden = torch.cat([elec_hid, chem_hid], dim=-1)
    output = self.linear(hidden)
    return output