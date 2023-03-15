from train._pkg import *
from govfunc._utils import *


def train(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    no_grad: bool = False,
) -> dict:
    """Train a model.

    Train a model (for 1 epoch) to predict the residual neural
    activity given a data loader of neural activity training set
    sequences.

    Args:
        loader: training set dataloader
        model: instance of a Network
        mask: selects indices of named neurons with data
        optimizer: gradient descent optimizer loaded with model params

    Returns:
        losses: dict w/ keys train_loss and base_train_loss
    """
    # Set model to train mode.
    model.train()
    criterion = model.loss_fn()
    base_loss, train_loss = 0, 0
    num_train_samples = 0
    # Iterate in batches over the training dataset.
    for i, data in enumerate(loader):
        X_train, Y_train, metadata = data  # X, Y: (batch_size, seq_len, num_neurons)
        X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)
        # Turn on gradients on tensors.
        X_train.requires_grad = True
        Y_train.requires_grad = True
        # Clear optimizer gradients.
        optimizer.zero_grad()
        # Baseline: loss if the model predicted the residual to be 0.
        base = criterion(X_train[:,:,mask], Y_train[:,:,mask])
        # Train
        Y_tr = model(X_train * mask.double())  # Forward pass.
        # Register hook.
        Y_tr.retain_grad()
        Y_tr.register_hook(lambda grad: grad * mask.double())
        # Compute training loss.
        loss = criterion(Y_tr[:,:,mask], Y_train[:,:,mask])
        loss.backward(retain_graph=True)  # Derive gradients.
        # # Prevent update of weights from neurons without data.
        # model.linear.weight.grad *= mask.unsqueeze(-1)
        # No backprop on epoch 0.
        if no_grad:
            optimizer.zero_grad()
        optimizer.step()  # Update parameters based on gradients.
        # Store train and baseline loss.
        base_loss += base.detach().item()
        train_loss += loss.detach().item()
        num_train_samples += X_train.size(0)
    # Average train and baseline losses.
    losses = {
        "train_loss": train_loss / (i + 1),
        "base_train_loss": base_loss / (i + 1),
        "num_train_samples": num_train_samples,
    }
    # Return losses.
    return losses


@torch.no_grad()
def test(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    mask: torch.Tensor,
) -> dict:
    """Evaluate a model.

    Test a model's residual neural activity prediction
    given a data loader of neural activity validation set
    sequences.

    Args:
        loader: test/validation set dataloader
        model: instance of a NetworkLSTM
        mask: mask which neurons in the dataset have real data

    Returns:
        losses: dict w/ keys test_loss and base_test_loss
    """
    # Set model to inference mode.
    model.eval()
    criterion = model.loss_fn()
    base_loss, test_loss = 0, 0
    num_test_samples = 0
    # Iterate in batches over the validation dataset.
    for i, data in enumerate(loader):
        X_test, Y_test, metadata = data  # X, Y: (batch_size, seq_len, num_neurons)
        X_test, Y_test = X_test.to(DEVICE), Y_test.to(DEVICE)
        # Baseline: loss if the model predicted the residual to be 0.
        base = criterion(X_test[:,:,mask], Y_test[:,:,mask])
        # Test
        Y_te = model(X_test * mask.double())  # Forward pass.
        loss = criterion(Y_te[:,:,mask], Y_test[:,:,mask])  # Compute the validation loss.
        # Store test and baseline loss.
        base_loss += base.detach().item()
        test_loss += loss.detach().item()
        num_test_samples += X_test.size(0)
    # Average test and baseline losses.
    losses = {
        "test_loss": test_loss / (i + 1),
        "base_test_loss": base_loss / (i + 1),
        "num_test_samples": num_test_samples,
    }
    # Return losses
    return losses


def split_train_test(
    data: torch.Tensor,
    k_splits: int = 2,
    seq_len: int = 101,
    batch_size: int = 32,
    train_size: int = 1024,
    test_size: int = 1024,
    shuffle: bool = True,
    reverse: bool = True,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Create neural activity train and test datasets.
    Returns train and test data loaders and masks.
    """
    # argument checking
    assert isinstance(k_splits, int) and k_splits > 1, "Ensure that `k_splits` > 1."
    assert isinstance(seq_len, int) and seq_len < len(
        data
    ), "Invalid `seq_len` entered."
    # split dataset into train and test sections
    chunk_size = len(data) // k_splits
    split_datasets = torch.split(data, chunk_size, dim=0)  # length k_splits tuple
    # create train and test masks. important that this is before dropping last split
    train_mask = torch.cat(
        [
            # split as train-test-train-test- ...
            (True if i % 2 == 0 else False) * torch.ones(len(section), dtype=torch.bool)
            for i, section in enumerate(split_datasets)
        ]
    )
    test_mask = ~train_mask
    # drop last split if too small. important that this is after making train/test masks
    tau = 1
    if (2 * tau >= len(split_datasets[-1])) or (len(split_datasets[-1]) < seq_len):
        split_datasets = split_datasets[:-1]
    train_splits = split_datasets[::2]
    test_splits = split_datasets[1::2]
    # make datasets; TODO: Parallelize this with `multiprocess.Pool`.
    # train dataset
    train_datasets = [
        NeuralActivityDataset(
            data,
            seq_len=seq_len,
            # keep per worm train size constant
            num_samples=train_size // len(train_splits),
            reverse=reverse,
        )
        for data in train_splits
    ]
    train_dataset = ConcatDataset(train_datasets)
    # tests dataset
    test_datasets = [
        NeuralActivityDataset(
            data,
            seq_len=seq_len,
            # keep per worm test size constant
            num_samples=test_size // len(test_splits),
            reverse=reverse,
        )
        for data in test_splits
    ]
    test_dataset = ConcatDataset(test_datasets)
    # make data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )
    # return data loaders and masks
    return train_loader, test_loader, train_mask, test_mask


def optimize_model(
    data: torch.Tensor,
    model: torch.nn.Module,
    mask: Union[torch.tensor, None] = None,
    optimizer: Union[torch.optim.Optimizer, None] = None,
    start_epoch: int = 1,
    learn_rate: float = 0.01,
    num_epochs: int = 100,
    **kwargs,
) -> tuple[torch.nn.Module, dict]:
    """
    Creates train and test data loaders from the given dataset
    and an optimizer given the model. Trains and validates the
    model for specified number of epochs. Returns the trained
    model and a dictionary with log information including losses.
    kwargs:  {
                k_splits: int,
                seq_len: int,
                batch_size: int,
                train_size: int,
                test_size: int,
                shuffle: bool,
                reverse: bool,
            }
    """
    # create the feature mask
    if mask is None:
        mask = torch.ones(NUM_NEURONS, dtype=torch.bool)
    assert mask.size(0) == NUM_NEURONS and mask.dtype == torch.bool
    mask = mask.to(DEVICE)
    # put model on device
    model = model.to(DEVICE)
    # create optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    # create data loaders and train/test masks
    train_loader, test_loader, train_mask, test_mask = split_train_test(data, **kwargs)
    # create log dictionary to return
    log = {
        "epochs": [],
        "base_train_losses": [],
        "base_test_losses": [],
        "train_losses": [],
        "test_losses": [],
        "num_train_samples": [],
        "num_test_samples": [],
        "centered_train_losses": [],
        "centered_test_losses": [],
        "model_state_dicts": [],
        "optimizer_state_dicts": [],
        "train_mask": train_mask,
        "test_mask": test_mask,
    }
    # iterate over the training data multiple times
    for epoch in range(start_epoch, num_epochs + start_epoch):
        # train the model
        train_log = train(train_loader, model, mask, optimizer, no_grad=(epoch == 0))
        test_log = test(test_loader, model, mask)
        base_train_loss, train_loss, num_train_samples = (
            train_log["base_train_loss"],
            train_log["train_loss"],
            train_log["num_train_samples"],
        )
        base_test_loss, test_loss, num_test_samples = (
            test_log["base_test_loss"],
            test_log["test_loss"],
            test_log["num_test_samples"],
        )
        centered_train_loss = train_loss - base_train_loss
        centered_test_loss = test_loss - base_test_loss
        if (num_epochs < 10) or (epoch % (num_epochs // 10) == 0):
            print(
                f"Epoch: {epoch:03d}, Train Loss: {centered_train_loss:.4f}, Val. Loss: {centered_test_loss:.4f}",
                end="\n\n",
            )
            # save epochs
            log["epochs"].append(epoch)
            # save losses and batch counts
            log["base_train_losses"].append(base_train_loss)
            log["base_test_losses"].append(base_test_loss)
            log["train_losses"].append(train_loss)
            log["test_losses"].append(test_loss)
            log["num_train_samples"].append(num_train_samples)
            log["num_test_samples"].append(num_test_samples)
            log["centered_train_losses"].append(centered_train_loss)
            log["centered_test_losses"].append(centered_test_loss)
    # return optimized model
    return model, log


def make_predictions(
    model: torch.nn.Module,
    dataset: dict,
    log_dir: str,
) -> None:
    """Make predicitons on a dataset with a trained model.

    Saves in the provdied log directory a .csv file for each of the following:
        * calcium neural activty
        * target calcium residuals
        * predicted calcium residuals
    Each .csv file has a column for each named neuron in the dataset plus two
    additional columns for the train mask and test mask respectively.

    Args:
        model: torch.nn.Module, Trained model.
        dataset: dict, Multi-worm dataset.

    Returns:
        None.
    """
    for worm, single_worm_dataset in dataset.items():
        os.makedirs(os.path.join(log_dir, worm), exist_ok=True)
        # get data to save
        named_neuron_to_idx = single_worm_dataset["named_neuron_to_idx"]
        calcium_data = single_worm_dataset["calcium_data"]
        named_neurons_mask = single_worm_dataset["named_neurons_mask"]
        train_mask = single_worm_dataset["train_mask"]
        labels = np.expand_dims(np.where(train_mask, "train", "test"), axis=-1)
        columns = list(named_neuron_to_idx) + ["train_test_label"]
        # make predictions with final model
        targets, predictions = model_predict(model, calcium_data)
        # save dataframes
        data = calcium_data[:, named_neurons_mask].numpy()
        data = np.hstack((data, labels))
        pd.DataFrame(data=data, columns=columns).to_csv(
            os.path.join(log_dir, worm, "ca_activity.csv"),
            index=True,
            header=True,
        )
        data = targets[:, named_neurons_mask].numpy()
        data = np.hstack((data, labels))
        pd.DataFrame(data=data, columns=columns).to_csv(
            os.path.join(log_dir, worm, "target_ca.csv"),
            index=True,
            header=True,
        )
        data = predictions[:, named_neurons_mask].numpy()
        data = np.hstack((data, labels))
        pd.DataFrame(data=data, columns=columns).to_csv(
            os.path.join(log_dir, worm, "predicted_ca.csv"),
            index=True,
            header=True,
        )
    return None


def model_predict(
    model: torch.nn.Module, calcium_data: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Makes predictions for all neurons in the
    calcium data tensor using a trained model.
    """
    model = model.double()
    model.eval()
    # model in/out
    calcium_data = calcium_data.squeeze()
    assert (
        calcium_data.ndim == 2 and calcium_data.size(1) == NUM_NEURONS
    ), "Calcium data has incorrect shape!"
    # get input and output
    input = calcium_data.to(DEVICE)
    output = model(input.unsqueeze(0)).squeeze()
    # targets/predictions
    targets = input.detach().cpu()
    predictions = output.detach().cpu()
    return targets, predictions


def gnn_train_val_mask(graph, train_ratio=0.7, train_mask=None):
    """
    Mutates a C. elegans connectome graph with injected data
    to include a training and validation mask.
    Returns the graph with `train_mask` and `val_mask` masks added
    as attributes.
    """
    # create the train and validation masks
    if train_mask is not None:
        assert (
            train_mask.ndim == 1 and train_mask.size(0) == graph.num_nodes
        ), "Invalid train_mask provided."
    else:
        train_mask = torch.rand(graph.num_nodes) < train_ratio
    val_mask = ~train_mask
    # make the train and test masks attributes of the data graph
    graph.train_mask = train_mask
    graph.val_mask = val_mask
    # put graph on GPU
    graph = graph.to(DEVICE)
    # return the graph with train and validation masks
    return graph


def gnn_train(loader, model, graph, optimizer):
    """Train a model given a dataset.
    Args:
        loader: training set dataloader
        model: instance of a Pytorch GNN model
        graph: subgraph of connectome with labelled data
        optimizer: gradient descent optimization
    Returns:
        train_loss, base_loss: traning loss and baseline
    """
    model.train()
    criterion = torch.nn.MSELoss()
    base_loss = 0
    train_loss = 0
    mask = graph.train_mask  # training mask
    graph = graph.subgraph(mask)
    X, y, _ = next(iter(loader))
    X, y = X[:, :, mask], y[:, :, mask]
    B = X.shape[0]
    for b in range(B):  # Iterate in batches over the training dataset.
        Xtr, ytr = X[b].T, y[b].T
        # Perform a single forward pass.
        out = model(Xtr, graph.edge_index, graph.edge_attr)
        # Compute the baseline loss
        base = criterion(Xtr, ytr)  # loss if model predicts y(t) for y(t+1)
        # Compute the training loss.
        loss = criterion(out, ytr)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        # Store train and baseline loss.
        base_loss += base.detach().item()
        train_loss += loss.detach().item()
    # return mean training and baseline losses
    return train_loss / B, base_loss / B


def gnn_test(loader, model, graph):
    """Evaluate a model on a given dataset.
        loader: validation set dataloader
        model: instance of a Pytorch GNN model
        graph: subgraph of connectome with labelled data
    Returns:
        val_loss, base_loss: validation loss and baseline
    """
    model.eval()
    criterion = torch.nn.MSELoss()
    base_loss = 0
    val_loss = 0
    mask = graph.val_mask  # validation mask
    graph = graph.subgraph(mask)
    X, y, _ = next(iter(loader))
    X, y = X[:, :, mask], y[:, :, mask]
    B = X.shape[0]
    for b in range(B):  # Iterate in batches over the test dataset.
        Xte, yte = X[b].T, y[b].T
        # Perform a single forward pass.
        out = model(Xte, graph.edge_index, graph.edge_attr)
        # Compute the baseline loss.
        base = criterion(Xte, yte)  # loss if model predicts y(t) for y(t+1)
        # Store the validation and baseline loss.
        base_loss += base.detach().item()
        val_loss += criterion(out, yte).detach().item()
    # return mean validation and baseline losses
    return val_loss / B, base_loss / B


def gnn_optimize_model(task, model):
    """
    Args:
        task: instance of GraphTask containing test data
        model: instance of a Pytorch model
    Returns:
        model: trained model
        log: log of train and test
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    log = {"epochs": [], "test_losses": [], "train_losses": []}
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
        log["epochs"].append(epoch)
        log["train_losses"].append(train_loss.item())
        log["test_losses"].append(test_loss.item())
    return model, log


def gnn_model_predict(task, model):
    """
    Have the GNN model to make predictions.
    Args:
        task: instance of GraphTask containing full dataset
        model: instance of a trained Pytorch model
    Returns:
        preds: (neurons, time) np.ndarray, model predictions
    """
    dataset = task()
    preds = np.empty((task.node_count, task.dataset_size))
    for time, snapshot in enumerate(dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        preds[:, [time]] = y_hat.clone().detach().numpy()
    return preds
