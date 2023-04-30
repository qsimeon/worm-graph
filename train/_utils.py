from train._pkg import *
from govfunc._utils import *


def train(
    loader: Union[list[torch.utils.data.DataLoader], torch.utils.data.DataLoader],
    model: torch.nn.Module,
    mask: Union[list[torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    no_grad: bool = False,
    use_residual: bool = False,
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
    # create a list of loaders and masks if only one
    if isinstance(loader, torch.utils.data.DataLoader):
        loaders = [loader]
        masks = [mask]
    else:  # list
        loaders = loader
        masks = mask
    # Set model to train mode.
    model.train()
    criterion = model.loss_fn()
    base_loss, train_loss = 0, 0
    num_train_samples = 0
    # Iterate in batches over the training dataset.
    i = 0
    # each worm in a cohort has its own dataloader
    for loader, mask in zip(loaders, masks):
        # each data loader has samples from a single worm
        for data in loader:
            i += 1
            (
                X_train,
                Y_train,
                metadata,
            ) = data  # X, Y: (batch_size, seq_len, num_neurons)
            X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)
            tau = metadata["tau"].detach()[0]  # penalize longer delays
            # Clear optimizer gradients.
            optimizer.zero_grad()
            # Baseline: loss if the model predicted value at next timestep equal to current value.
            base = criterion(
                (0 if use_residual else 1) * X_train[:, :, mask], Y_train[:, :, mask]
            )
            # Train
            Y_tr = model(X_train * mask, tau)  # Forward pass.
            # Compute training loss.
            loss = criterion(Y_tr[:, :, mask], Y_train[:, :, mask])
            loss.backward()  # Derive gradients.
            # No backprop on epoch 0.
            if no_grad:
                optimizer.zero_grad()
            optimizer.step()  # Update parameters based on gradients.
            # Store train and baseline loss.
            base_loss += base.detach().item()
            train_loss += loss.detach().item()
            num_train_samples += X_train.detach().size(0)
    # Average train and baseline losses.
    losses = {
        "train_loss": train_loss / (i + 1),
        "base_train_loss": base_loss / (i + 1),
        "centered_train_loss": (train_loss - base_loss) / (i + 1),
        "num_train_samples": num_train_samples,
    }
    # Return losses.
    return losses


@torch.no_grad()
def test(
    loader: Union[list[torch.utils.data.DataLoader], torch.utils.data.DataLoader],
    model: torch.nn.Module,
    mask: Union[list[torch.Tensor], torch.Tensor],
    use_residual: bool = False,
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
    # create a list of loaders and masks if only one is given
    if isinstance(loader, torch.utils.data.DataLoader):
        loaders = [loader]
        masks = [masks]
    else:  # list
        loaders = loader
        masks = mask
    # Set model to inference mode.
    model.eval()
    criterion = model.loss_fn()
    base_loss, test_loss = 0, 0
    num_test_samples = 0
    # Iterate in batches over the validation dataset.
    i = 0
    # each worm in cohort has its own dataloader
    for loader, mask in zip(loaders, masks):
        # each data loader has samples from a single worm
        for data in loader:
            i += 1
            X_test, Y_test, metadata = data  # X, Y: (batch_size, seq_len, num_neurons)
            X_test, Y_test = X_test.to(DEVICE), Y_test.to(DEVICE)
            tau = metadata["tau"][0]  # penalize longer delays
            # Baseline: loss if the model predicted value at next timestep equal to current value.
            base = criterion(
                (0 if use_residual else 1) * X_test[:, :, mask], Y_test[:, :, mask]
            )
            # Test
            Y_te = model(X_test * mask, tau)  # Forward pass.
            # Compute the validation loss.
            loss = criterion(Y_te[:, :, mask], Y_test[:, :, mask])
            # Store test and baseline loss.
            base_loss += base.detach().item()
            test_loss += loss.detach().item()
            num_test_samples += X_test.detach().size(0)
    # Average test and baseline losses.
    losses = {
        "test_loss": test_loss / (i + 1),
        "base_test_loss": base_loss / (i + 1),
        "centered_test_loss": (test_loss - base_loss) / (i + 1),
        "num_test_samples": num_test_samples,
    }
    # Return losses
    return losses


def split_train_test(
    data: torch.Tensor,
    k_splits: int = 2,
    seq_len: Union[int, list[int]] = 100,  # TODO: allow a list of `seq_len`s
    num_samples: int = 10,
    num_batches: int = 2,
    time_vec: Union[torch.Tensor, None] = None,
    shuffle: bool = True,
    reverse: bool = True,
    tau: Union[int, list[int]] = 1,
    use_residual: bool = False,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Create the train and test dataloaders and masks
    for neural activity from a single worm.
    """
    # argument checking
    assert isinstance(k_splits, int) and k_splits > 1, "Ensure that `k_splits` > 1."
    if isinstance(seq_len, int):
        assert seq_len < len(data), "Invalid `seq_len` integer entered."
        seq_len = [seq_len]  # convert`seq_len` to a list
    else:
        seq_len = list(seq_len)
        assert isinstance(seq_len, list) and all(
            [0 < sl < len(data) for sl in seq_len]
        ), "Invalid `seq_len` list entered."
    if isinstance(tau, int):
        assert all(
            tau < len(data) - sl for sl in seq_len
        ), "Invalid `tau` integer entered."
        tau = [tau]  # convert`tau` to a list
    else:
        tau = list(tau)
        assert isinstance(tau, list) and all(
            [0 < t < len(data) - sl for t in tau for sl in seq_len]
        ), "Invalid `tau` list entered."
    # make time vector
    if time_vec is None:
        time_vec = torch.arange(len(data), dtype=torch.float32)
    else:
        time_vec = time_vec.to(torch.float32)
    assert torch.is_tensor(time_vec) and len(time_vec) == len(
        data
    ), "Enter a time vector with same length as data."
    # detach computation graph from tensors
    time_vec = time_vec.detach()
    data = data.detach()
    # split dataset into train and test sections
    chunk_size = len(data) // k_splits
    split_datasets = torch.split(data, chunk_size, dim=0)  # length k_splits list
    split_times = torch.split(time_vec, chunk_size, dim=0)
    # create train and test masks. important that this is before dropping last split
    train_mask = torch.cat(
        [
            # split as train-test-train-test- ...
            (True if i % 2 == 0 else False) * torch.ones(len(section), dtype=torch.bool)
            for i, section in enumerate(split_datasets)
        ]
    ).detach()
    test_mask = ~train_mask.detach()
    # drop last split if too small. important that this is after making train/test masks
    len_last = len(split_datasets[-1])
    if any(t >= len_last // 2 for t in tau) or any(len_last < sl for sl in seq_len):
        split_datasets = split_datasets[:-1]
        split_times = split_times[:-1]
    # make dataset splits
    train_splits, train_times = split_datasets[::2], split_times[::2]
    test_splits, test_times = split_datasets[1::2], split_times[1::2]
    # pick a random `tau` and `seq_len` from the lists
    _tau = random.choice(tau)
    _seq_len = random.choice(seq_len)
    # calculate number of train/ test samples per split
    train_size_per_split = (num_samples // len(train_splits)) + (
        num_samples % len(train_splits)
    )
    test_size_per_split = (num_samples // len(test_splits)) + (
        num_samples % len(test_splits)
    )
    # train dataset
    train_datasets = [
        NeuralActivityDataset(
            _data.detach(),
            seq_len=_seq_len,
            num_samples=train_size_per_split,
            reverse=reverse,
            time_vec=train_times[i],
            tau=_tau,
            use_residual=use_residual,
        )
        for i, _data in enumerate(train_splits)
    ]
    train_dataset = ConcatDataset(train_datasets)
    # test dataset
    test_datasets = [
        NeuralActivityDataset(
            _data.detach(),
            seq_len=_seq_len,
            num_samples=test_size_per_split,
            reverse=(not reverse),
            time_vec=test_times[i],
            tau=_tau,
            use_residual=use_residual,
        )
        for i, _data in enumerate(test_splits)
    ]
    test_dataset = ConcatDataset(test_datasets)
    # calculate the batch size
    batch_size = max(1, num_samples // num_batches)
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
    # return datasets and masks
    return train_loader, test_loader, train_mask, test_mask


def optimize_model(
    model: torch.nn.Module,
    train_loader: Union[list[torch.utils.data.DataLoader], torch.utils.data.DataLoader],
    test_loader: Union[list[torch.utils.data.DataLoader], torch.utils.data.DataLoader],
    neurons_mask: Union[list[Union[torch.Tensor, None]], torch.Tensor, None] = None,
    optimizer: Union[torch.optim.Optimizer, None] = None,
    start_epoch: int = 1,
    learn_rate: float = 0.01,
    num_epochs: int = 1,
    use_residual: bool = False,
) -> tuple[torch.nn.Module, dict]:
    """
    Creates train and test data loaders from the given dataset
    and an optimizer given the model. Trains and validates the
    model for specified number of epochs. Returns the trained
    model and a dictionary with log information including losses.
    """
    # check the train and test loader input
    assert isinstance(
        train_loader, (list, torch.utils.data.DataLoader)
    ), "Wrong input type for `train_loader`."
    assert isinstance(
        test_loader, (list, torch.utils.data.DataLoader)
    ), "Wrong input type for `test_loader`."
    if isinstance(train_loader, list):
        batch_in, _, _ = next(iter(train_loader[0]))
    elif isinstance(train_loader, torch.utils.data.DataLoader):
        batch_in, _, _ = next(iter(train_loader))
    NUM_NEURONS = batch_in.size(-1)
    # create the neurons/feature mask
    if isinstance(neurons_mask, torch.Tensor):
        assert (
            neurons_mask.size(0) == NUM_NEURONS and neurons_mask.dtype == torch.bool
        ), "Please use a valid boolean mask for neurons."
        neurons_mask = neurons_mask.detach().to(DEVICE)
    elif isinstance(neurons_mask, list):
        for i, mask in enumerate(neurons_mask):
            assert (
                mask.size(0) == NUM_NEURONS and mask.dtype == torch.bool
            ), "Please use a valid boolean mask for neurons."
            neurons_mask[i] = neurons_mask[i].detach().to(DEVICE)
    else:
        neurons_mask = torch.ones(NUM_NEURONS, dtype=torch.bool).detach().to(DEVICE)
    # put model on device
    model = model.to(DEVICE)
    # create optimizer
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    # create log dictionary to return
    log = {
        "epochs": np.zeros(num_epochs, dtype=int),
        "base_train_losses": np.zeros(num_epochs, dtype=np.float32),
        "base_test_losses": np.zeros(num_epochs, dtype=np.float32),
        "train_losses": np.zeros(num_epochs, dtype=np.float32),
        "test_losses": np.zeros(num_epochs, dtype=np.float32),
        "num_train_samples": np.zeros(num_epochs, dtype=int),
        "num_test_samples": np.zeros(num_epochs, dtype=int),
        "centered_train_losses": np.zeros(num_epochs, dtype=np.float32),
        "centered_test_losses": np.zeros(num_epochs, dtype=np.float32),
    }
    # iterate over the training data for `num_epochs` (usually 1)
    iter_range = range(start_epoch, num_epochs + start_epoch)
    for i, epoch in enumerate(iter_range):
        # train and validate the model
        # TODO: how can I run the train and test loops in parallel?
        train_log = train(
            train_loader,
            model,
            neurons_mask,
            optimizer,
            no_grad=(epoch == 0),
            use_residual=use_residual,
        )
        test_log = test(
            test_loader,
            model,
            neurons_mask,
            use_residual=use_residual,
        )
        # retrieve losses
        centered_train_loss, base_train_loss, train_loss, num_train_samples = (
            train_log["centered_train_loss"],
            train_log["base_train_loss"],
            train_log["train_loss"],
            train_log["num_train_samples"],
        )
        centered_test_loss, base_test_loss, test_loss, num_test_samples = (
            test_log["centered_test_loss"],
            test_log["base_test_loss"],
            test_log["test_loss"],
            test_log["num_test_samples"],
        )
        # save epochs, losses and batch counts
        log["epochs"][i] = epoch
        log["base_train_losses"][i] = base_train_loss
        log["base_test_losses"][i] = base_test_loss
        log["train_losses"][i] = train_loss
        log["test_losses"][i] = test_loss
        log["num_train_samples"][i] = num_train_samples
        log["num_test_samples"][i] = num_test_samples
        log["centered_train_losses"][i] = centered_train_loss
        log["centered_test_losses"][i] = centered_test_loss
        # print to standard output
        if (num_epochs < 10) or (epoch % (num_epochs // 10) == 0):
            print(
                f"\nEpoch: {epoch:03d}, Train Loss: {centered_train_loss:.4f}, Val. Loss: {centered_test_loss:.4f}",
                end="\n\n",
            )
    # return optimized model
    return model, log


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
        preds[:, [time]] = y_hat.detach().numpy()
    return preds
