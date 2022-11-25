import torch
import numpy as np
from tqdm import tqdm

def train(dataset, model):
    """Train a model given a dataset.
    Args:
        dataset: the training data
        model: instance of a Pytorch GNN model

    Returns:
         loss: training loss
    """
    model.train()
    loss = 0
    for time, snapshot in enumerate(dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss = loss + torch.mean((y_hat - snapshot.y) ** 2)
    loss = loss / (time + 1)
    return loss

def test(dataset, model):
    """Evaluate a model on a given dataset.
    Args:
        dataset: the test data
        model: instance of a Pytorch model
    Returns:
        loss: validation loss
    """
    model.eval()
    loss = 0
    for time, snapshot in enumerate(dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss = loss + torch.mean((y_hat-snapshot.y)**2)  
    loss = loss / (time+1)
    return loss

def optimize_model(task, model):
    """
    Args:
        task: instance of GraphTask containing test data
        model: instance of a Pytorch model
    Returns:
        model: trained model
        log: log of train and test
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    log = {'epochs': [], 'test_losses': [], 'train_losses': []}
    train_dataset, test_dataset = task.train_test_split()

    for epoch in tqdm(range(20)):
        # forward pass
        train_loss = train(train_dataset, model)
        # backpropagation
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # validation
        test_loss = test(test_dataset, model)
        # logging
        log['epochs'].append(epoch)
        log['train_losses'].append(train_loss.item())
        log['test_losses'].append(test_loss.item())
    return model, log


def model_predict(task, model):
    """Ask the model to make predictions.
    Args:
        task: instance of GraphTask containing full dataset
        model: instance of a trained Pytorch model
    Returns:
        preds: (neurons, time) np.ndarray, model predictions
    """
    dataset = task()
    preds = np.empty((task.node_count, task.data_size))
    for time, snapshot in enumerate(dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        preds[:, [time]] = y_hat.clone().detach().numpy()
    return preds