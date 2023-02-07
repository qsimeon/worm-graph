import torch
from utils import DEVICE as device
from data.map_dataset import MapDataset
from data.batch_sampler import BatchSampler

# @title Make a Pytorch-style train and test pipeline.
# @markdown This pipeline will be used by all models.
# @markdown All models should include a method `loss_fn()` that specifies the
# @markdown loss function to be used with the model.


def train(loader, model, mask, optimizer, no_grad=False):
    """
    Train a model given a dataset for a single epoch.
      Args:
          loader: training set dataloader
          model: instance of a NetworkLSTM
          mask: mask which neurons in the dataset have real data
          optimizer: gradient descent optimizer with model params on it
      Returns:
          losses: dict w/ keys train_loss and base_train_loss
    """
    # set model to train
    model.train()
    criterion = model.loss_fn()
    base_loss, train_loss = 0, 0
    # Iterate in batches over the training dataset.
    for i, data in enumerate(loader):
        X_train, Y_train, meta = data  # (batch_size, seq_len, num_neurons)
        tau = meta["tau"][0]
        optimizer.zero_grad()  # Clear gradients.
        # Baseline: loss if the model predicted the residual to be 0
        base = criterion(
            torch.zeros_like(Y_train[:, :, mask]), (Y_train - X_train)[:, :, mask]
        ) / (tau + 1)
        # Train
        Y_tr = model(X_train)  # Forward pass.
        Y_tr.retain_grad()
        Y_tr.register_hook(lambda grad: grad * mask.double())
        loss = criterion(Y_tr[:, :, mask], (Y_train - X_train)[:, :, mask]) / (
            tau + 1
        )  # Compute training loss.
        loss.backward()  # Derive gradients.
        if no_grad:
            optimizer.zero_grad()
        optimizer.step()  # Update parameters based on gradients.
        # Store train and baseline loss.
        base_loss += base.detach().item()
        train_loss += loss.detach().item()
    # Average train and baseline losses
    losses = {
        "train_loss": train_loss / (i + 1),
        "base_train_loss": base_loss / (i + 1),
    }
    # return mean train and baseline losses
    return losses


@torch.no_grad()
def test(loader, model, mask):
    """
    Evaluate a model on a given dataset.
        loader: test/validation set dataloader
        model: instance of a NetworkLSTM
        mask: mask which neurons in the dataset have real data
    Returns:
        losses: dict w/ keys test_loss and base_test_loss
    """
    model.eval()  # this turns of grad
    criterion = model.loss_fn()
    base_loss, test_loss = 0, 0
    # Iterate in batches over the validation dataset.
    for i, data in enumerate(loader):
        X_test, Y_test, meta = data  # (batch_size, seq_len, num_neurons)
        tau = meta["tau"][0]
        # Baseline: loss if the model predicted the residual to be 0
        base = criterion(
            torch.zeros_like(Y_test[:, :, mask]), (Y_test - X_test)[:, :, mask]
        ) / (tau + 1)
        # Test
        Y_te = model(X_test)  # Forward pass.
        loss = criterion(Y_te[:, :, mask], (Y_test - X_test)[:, :, mask]) / (
            tau + 1
        )  # Compute the validation loss.
        # Store test and baseline loss.
        base_loss += base.detach().item()
        test_loss += loss.detach().item()
    # Average test and baseline losses
    losses = {"test_loss": test_loss / (i + 1), "base_test_loss": base_loss / (i + 1)}
    return losses


def optimize_model(
    dataset, model, mask=None, optimizer=None, num_epochs=100, seq_len=1, data_size=1000
):
    """
    Creates train and test loaders given a task/dataset.
    Creates the optimizer given the model.
    Trains and validates the model for specified number of epochs.
    Returns a dict of epochs, and train, test and baseline losses.
    """
    # create the mask
    if mask is None:
        mask = torch.ones(302, dtype=torch.bool)
    assert mask.size(0) == 302 and mask.dtype == torch.bool
    mask.requires_grad = False
    mask = mask.to(device)
    # put model on device
    model = model.to(device)
    # create optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # create MapDatasets
    max_time = len(dataset)
    train_dataset = MapDataset(
        dataset[: max_time // 2],
        tau=1,
        seq_len=seq_len,
        increasing=False,
        reverse=True,
        size=data_size,
    )
    test_dataset = MapDataset(
        dataset[max_time // 2 :],
        tau=1,
        seq_len=seq_len,
        increasing=False,
        reverse=True,
        size=2048,
    )  # fix test size
    # create train and test loaders
    train_sampler = BatchSampler(train_dataset.batch_indices)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_sampler
    )
    test_sampler = BatchSampler(test_dataset.batch_indices)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_sampler)
    # create log dictionary to return
    log = {
        "base_train_losses": [],
        "base_test_losses": [],
        "train_losses": [],
        "test_losses": [],
        "epochs": [],
    }
    log.update({"data_size": train_dataset.size, "seq_len": seq_len})
    # iterate over the training data multiple times
    for epoch in range(num_epochs + 1):
        # train the model
        train_log = train(train_loader, model, mask, optimizer, no_grad=(epoch == 0))
        test_log = test(test_loader, model, mask)
        base_train_loss, train_loss = (
            train_log["base_train_loss"],
            train_log["train_loss"],
        )
        base_test_loss, test_loss = test_log["base_test_loss"], test_log["test_loss"]
        if epoch % (num_epochs // 100) == 0:
            print(
                f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val. Loss: {test_loss:.4f}"
            )
            log["epochs"].append(epoch)
            log["base_train_losses"].append(base_train_loss)
            log["base_test_losses"].append(base_test_loss)
            log["train_losses"].append(train_loss)
            log["test_losses"].append(test_loss)
    # return optimized model
    return model, log


def model_predict(named_calcium_data, model):
    """
    Makes predictions for all neurons in the given
    worm dataset using a trained model.
    """
    # model in/out
    input = named_calcium_data.squeeze().to(device)
    output = model(input)
    # targets/preds
    targets = (input[1:] - input[:-1]).detach().cpu()
    predictions = output[:-1].detach().cpu()
    return targets, predictions
