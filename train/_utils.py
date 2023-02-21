from train._pkg import *


def train(loader, model, mask, optimizer, no_grad=False):
    """
    Train (for 1 epoch) a model to predict the residual neural
    activity given a dataset of neural activity for 1 epoch.
      Args:
          loader: training set dataloader
          model: instance of a NetworkLSTM
          mask: selects indices of neurons in the dataset with data
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
        X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)
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
        X_test, Y_test = X_test.to(DEVICE), Y_test.to(DEVICE)
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


def split_train_test(
    data: torch.tensor,
    k_splits: int = 2,
    seq_len: Union[int, list] = 101,
    train_size: int = 1,
    test_size: int = 1,
    tau: int = 1,
    shuffle: bool = True,
    reverse: bool = True,
):
    """
    Create neural activity train and test datasets.
    Returns train and test data loaders and masks.
    TODO: Allow `split_train_test` to work with a list for `seq_len` to allow
    multiscale training.
    """
    # argument checking
    assert isinstance(k_splits, int) and k_splits > 1, "Ensure that k_splits > 1."
    # allow multi-scale sequence lengths
    if isinstance(seq_len, int):
        seq_len = [seq_len]
    seq_len = list(seq_len)
    # cannot shuffle if multi-scaling
    if len(seq_len) > 1:
        shuffle = False
    # split dataset into train and test sections
    chunk_size = len(data) // k_splits
    split_datasets = torch.split(data, chunk_size, dim=0)  # len k_splits tensor
    # create train and test masks. it is important that this is before dropping last split
    train_mask = torch.cat(
        [
            (True if i % 2 == 0 else False) * torch.ones(len(section), dtype=torch.bool)
            for i, section in enumerate(split_datasets)
        ]
    )
    test_mask = ~train_mask
    # drop last split if too small. important that this is after making train mask
    if (2 * tau >= len(split_datasets[-1])) or (len(split_datasets[-1]) < min(seq_len)):
        split_datasets = split_datasets[:-1]
    train_splits = split_datasets[::2]
    test_splits = split_datasets[1::2]
    # make datasets
    train_div = len(seq_len) * len(train_splits)
    train_datasets = [
        NeuralActivityDataset(
            dset,
            tau=tau,
            seq_len=seq,
            reverse=reverse,
            size=train_size // train_div,
        )
        for seq in seq_len
        for dset in train_splits
    ]
    test_div = len(seq_len) * len(test_splits)
    test_datasets = [
        NeuralActivityDataset(
            dset,
            tau=tau,
            seq_len=seq,
            reverse=reverse,
            size=test_size // test_div,
        )
        for seq in seq_len
        for dset in test_splits
    ]
    # batch indices
    train_indices = []
    test_indices = []
    prev_bn = 0
    for dset in train_datasets:
        train_indices.append(dset.batch_indices + prev_bn)
        prev_bn += dset.batch_indices[-1] + 1
    prev_bn = 0
    for dset in test_datasets:
        test_indices.append(dset.batch_indices + prev_bn)
        prev_bn += dset.batch_indices[-1] + 1
    # shuffle data
    if shuffle == True:
        train_shuffle_inds = random.sample(
            range(len(train_datasets)), k=len(train_datasets)
        )
        test_shuffle_inds = random.sample(
            range(len(test_datasets)), k=len(test_datasets)
        )
        train_datasets = [train_datasets[idx] for idx in train_shuffle_inds]
        test_datasets = [test_datasets[idx] for idx in test_shuffle_inds]
        train_indices = [train_indices[idx] for idx in train_shuffle_inds]
        test_indices = [test_indices[idx] for idx in test_shuffle_inds]
    # create the combined train and test datasets, samplers and data loaders
    train_dataset = ConcatDataset(train_datasets)
    train_sampler = BatchSampler(train_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        pin_memory=True,
    )
    test_dataset = ConcatDataset(test_datasets)
    test_sampler = BatchSampler(test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        pin_memory=True,
    )
    return train_loader, test_loader, train_mask, test_mask


def optimize_model(
    data: torch.tensor,
    model: torch.nn.Module,
    mask=None,
    optimizer=None,
    start_epoch=1,
    learn_rate=0.01,
    num_epochs=100,
    **kwargs,
):
    """
    Creates train and test data loaders from the given dataset
    and an optimizer given the model. Trains and validates the
    model for specified number of epochs. Returns the trained
    model and a dictionary with log information including losses.
    kwargs:  {
                k_splits: int,
                seq_len: int
                train_size: int,
                test_size: int,
                tau: int,
                shuffle: bool,
                reverse: bool,
            }
    """
    # create the feature mask
    if mask is None:
        mask = torch.ones(NUM_NEURONS, dtype=torch.bool)
    assert mask.size(0) == NUM_NEURONS and mask.dtype == torch.bool
    mask.requires_grad = False
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
        base_train_loss, train_loss = (
            train_log["base_train_loss"],
            train_log["train_loss"],
        )
        base_test_loss, test_loss = test_log["base_test_loss"], test_log["test_loss"]
        centered_train_loss = train_loss - base_train_loss
        centered_test_loss = test_loss - base_test_loss
        if (num_epochs < 10) or (epoch % (num_epochs // 10) == 0):
            print(
                f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val. Loss: {test_loss:.4f}",
                end="\n\n",
            )
            # save epochs
            log["epochs"].append(epoch)
            # save losses
            log["base_train_losses"].append(base_train_loss)
            log["base_test_losses"].append(base_test_loss)
            log["train_losses"].append(train_loss)
            log["test_losses"].append(test_loss)
            log["centered_train_losses"].append(centered_train_loss)
            log["centered_test_losses"].append(centered_test_loss)
    # return optimized model
    return model, log


def make_predictions(model, dataset, log_dir):
    """
    Make predicitons with trained model and save CSV files
    inside given log directory.
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
        data = torch.nn.functional.pad(
            targets[:, named_neurons_mask], (0, 0, 1, 0)
        ).numpy()
        data = np.hstack((data, labels))
        pd.DataFrame(data=data, columns=columns).to_csv(
            os.path.join(log_dir, worm, "target_ca_residual.csv"),
            index=True,
            header=True,
        )
        data = torch.nn.functional.pad(
            predictions[:, named_neurons_mask], (0, 0, 0, 1)
        ).numpy()
        data = np.hstack((data, labels))
        pd.DataFrame(data=data, columns=columns).to_csv(
            os.path.join(log_dir, worm, "predicted_ca_residual.csv"),
            index=True,
            header=True,
        )
    return None


def model_predict(model, calcium_data):
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
    input = calcium_data.to(DEVICE)
    # add batch dimension
    output = model(input.unsqueeze(0)).squeeze()
    # targets/predictions
    targets = (input[1:] - input[:-1]).detach().cpu()
    predictions = output[:-1].detach().cpu()
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
