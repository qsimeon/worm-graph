import dgl.function as fn
import matplotlib.pyplot as plt
import torch
import dgl
from torch import nn
import torch.nn.functional as F


from dataset import MineDataset

data_dir = '../data'

# Load dataset
dataset = MineDataset(
    data_dir=data_dir,
    input_steps=80,
    output_steps=1,
)


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })

        self.output_weights = nn.Linear(out_size, out_size)

    def forward(self, G, feats):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            #             neuron_basic chemical neuron_basic
            #             neuron_basic electric neuron_basic
            # Compute W_r * h
            Wh = self.weight[etype](feats)
            #             import pdb
            #             pdb.set_trace()
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_src('Wh_%s' % etype, 'm'), fn.sum('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return self.output_weights(G.nodes['neuron_basic'].data['h']) + Wh


class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size, num_layers=2):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embeds = nn.Parameter(torch.Tensor(279, in_size))
        nn.init.xavier_uniform_(embeds)
        self.embed = embeds

        node_features = G.nodes['neuron_basic'].data['feats']
        node_features = node_features.reshape(node_features.shape[0], -1)

        # create layers
        layers = []
        layers.append(
            HeteroRGCNLayer(in_size + (node_features.shape[-1]), hidden_size,
                            G.etypes))
        #         layers.append(nn.Linear(in_size+(node_features.shape[-1]), hidden_size)) for not considering graph

        for _ in range(num_layers):
            layers.append(HeteroRGCNLayer(hidden_size, hidden_size, G.etypes))
        #             layers.append(nn.Linear(hidden_size, hidden_size)) for not considering graph

        layers.append(HeteroRGCNLayer(hidden_size, out_size, G.etypes))
        #         layers.append(nn.Linear(hidden_size, out_size)) for not considering graph
        self.layers = nn.ModuleList(layers)

    def forward(self, G):
        node_features = G.nodes['neuron_basic'].data['feats']
        batch_size = node_features.shape[0] // (279)
        node_features = node_features.reshape(node_features.shape[0], -1)
        embed = torch.tile(self.embed, (batch_size, 1))

        h = torch.cat([embed, node_features], axis=-1)

        for i, layer in enumerate(self.layers[:-1]):
            h = layer(G, h)  # layer(h) for not considering graph
            h = F.leaky_relu(h)
        #             h = self.norms[i](h)

        h = self.layers[-1](G, h)
        h = h.reshape(h.shape, -1, 2)
        return h


# Create the model. The output has three logits for three classes.
model = HeteroRGCN(dataset[0], 10, 128, 1, num_layers=2)
model.train()

opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.0)  # 5e-4

best_loss = 1e10

losses = []
g = dgl.batch([g for g in dataset[:200]])
for epoch in range(5000):
    #     indices = np.random.choice(128)  if batch size is 128
    #     g = dgl.batch([g for g in dataset[indices]])

    voltages = model(g)
    labels = g.nodes['neuron_basic'].data['labels']  # [:,0]
    labels = labels.reshape(voltages.shape)
    # The loss is computed only for labeled nodes.
    loss = F.mse_loss(voltages, labels)

    if loss < best_loss:
        best_loss = loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 5 == 0:
        losses.append(loss.detach().cpu())

    if epoch % 1 == 0:
        print('Step: %d, Loss %.4f, Best Loss %.4f' % (
            epoch,
            loss.item(),
            best_loss.item(),
        ))

plt.plot(losses[100:])
