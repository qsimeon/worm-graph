import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric_temporal.nn.recurrent import TGCN, DCRNN
from torch_geometric_temporal.nn.recurrent import EvolveGCNH, GConvGRU


class ConvGNN(torch.nn.Module):
    def __init__(self):
        super(ConvGNN, self).__init__()


class ConvGNN_1(ConvGNN):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(ConvGNN, self).__init__()
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


class ConvGNN_2(ConvGNN):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(ConvGNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
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


class RGCN_1(RecurrentGCN):
    '''DCRNN'''
    def __init__(self, node_features):
        super(RGCN_1, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


class RGCN_2(RecurrentGCN):
    def __init__(self, node_count, node_features):
        '''EvolveGCNH'''
        super(RGCN_2, self).__init__()
        self.recurrent = EvolveGCNH(node_count, node_features)
        self.linear = torch.nn.Linear(node_features, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


class RGCN_3(RecurrentGCN):
    '''GConvGRU'''
    def __init__(self, node_features):
        super(RGCN_3, self).__init__()
        self.recurrent = GConvGRU(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h