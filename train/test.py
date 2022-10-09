import torch

def test(dataset, model):
    """Evaluate a model ona given dataset
    TODO: finish this.
    Args:
        dataset: dataset of validation samples
        model: instance of a Pytorch model

    Returns:
         cost: loss metric of the test data
    """
    model.eval()
    cost = 0

    for time, snapshot in enumerate(dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost = cost.item()
    
    return cost