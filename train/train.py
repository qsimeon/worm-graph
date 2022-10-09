from tqdm import tqdm
import torch

# 
def train(dataset, model):
    """Train a model given a dataset
    TODO: finish this.
    Args:
        dataset: dataset of training samples
        model: instance of a Pytorch model

    Returns:
         model: trained model
         log: log of training
    """
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for epoch in tqdm(range(200)):
        cost = 0
        for time, snapshot in enumerate(dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
        cost = cost / (time + 1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return model