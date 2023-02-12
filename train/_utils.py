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
    dataset,
    model,
    mask=None,
    optimizer=None,
    seq_len=1,
    start_epoch=1,
    learn_rate=0.01,
    num_epochs=100,
    dataset_size=1000,
):
    """
    Creates train and test data loaders from the given dataset
    and an optimizer given the model. Trains and validates the
    model for specified number of epochs. Returns the trained
    model and a dict of train, test and baseline losses.
    """
    # create the mask
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
    # create neural activity train and test datasets
    max_time = len(dataset)
    train_dataset = NeuralActivityDataset(
        dataset[: max_time // 2],
        tau=1,
        seq_len=seq_len,
        increasing=False,
        reverse=True,
        size=dataset_size,
    )
    test_dataset = NeuralActivityDataset(
        dataset[max_time // 2 :],
        tau=1,
        seq_len=seq_len,
        increasing=False,
        reverse=True,
        size=2048,
    )  # fixed test size
    # create train and test loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_dataset.batch_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_sampler=test_dataset.batch_sampler
    )
    # create log dictionary to return
    log = {
        "epochs": [],
        "base_train_losses": [],
        "base_test_losses": [],
        "train_losses": [],
        "test_losses": [],
        "model_state_dicts": [],
        "optimizer_state_dicts": [],
    }
    log.update({"dataset_size": train_dataset.size, "seq_len": seq_len})
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
        if epoch % (num_epochs // 10) == 0:
            print(
                f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val. Loss: {test_loss:.4f}",
                end="\n\n",
            )
            # saves epochs
            log["epochs"].append(epoch)
            # save losses
            log["base_train_losses"].append(base_train_loss)
            log["base_test_losses"].append(base_test_loss)
            log["train_losses"].append(train_loss)
            log["test_losses"].append(test_loss)
            # save checkpoints
            log["model_state_dicts"].append(model.state_dict())
            log["optimizer_state_dicts"].append(optimizer.state_dict())
    # return optimized model
    return model, log


def model_predict(calcium_data, model):
    """
    Makes predictions for all neurons in the
    calcium dataset using a trained model.
    """
    model.eval()
    # model in/out
    calcium_data = calcium_data.squeeze()
    assert (
        calcium_data.ndim == 2 and calcium_data.size(1) == NUM_NEURONS
    ), "Calcium data has incorrect shape!"
    input = calcium_data.to(DEVICE)
    output = model(input)
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


def lstm_hidden_size_experiment(
    dataset,
    num_epochs,
    input_size,
    num_layers=1,
    hid_mult=np.array([3, 2]),
    seq_len=range(1, 11, 1),
):
    """
    Helper function to experiment with different input sizes for the LSTM model.
    dataset: the dataset to train on.
    num_epochs: number of epochs to train for.
    input_size: number of input features (neurons).
    num_layers: number of hidden layers to use in the LSTM.
    hid_mult: np.array of integers to multiple input_size by.
    seq_len: array of sequnce lengths to train on.
    """
    hidden_experiment = dict()
    # we experiment with different hidden sizes
    for hidden_size in input_size * hid_mult:
        hidden_size = int(hidden_size)
        print("Hidden size: %d\n" + "~~~" * 10 % hidden_size, end="\n\n")
        # initialize model, optimizer and loss function
        lstm_model = NetworkLSTM(input_size, hidden_size, num_layers).double()
        # optimize the model
        lstm_model, log = optimize_model(
            dataset=dataset, model=lstm_model, num_epochs=num_epochs, seq_len=seq_len
        )
        # log results of this experiment
        hidden_experiment[hidden_size] = log
    return hidden_experiment


def more_data_training(
    model_class,
    single_worm_dataset,
    num_epochs=100,
    worm="worm**",
    model_name="",
    seq_len=1,
    plotting=False,
):
    """
    A function to investigate the effect of the amount of data
    (sequence length fixed) on the training and generalization of a model.
    """
    results_list = []
    # parse worm dataset
    # get the calcium data for this worm
    new_calcium_data = single_worm_dataset["named_data"]
    mask = single_worm_dataset["neurons_mask"]
    # get the neuron to idx map
    neuron_to_idx = single_worm_dataset["named_neuron_to_idx"]
    idx_to_neuron = dict((v, k) for k, v in neuron_to_idx.items())
    # pick a neuron
    neuron_idx = np.random.choice(list(neuron_to_idx.keys())) - 1
    neuron = neuron_to_idx[neuron_idx]
    # get max time and number of neurons
    max_time = single_worm_dataset["max_time"]
    num_neurons = single_worm_dataset["num_neurons"]
    # iterate over different dataset sizes
    data_sizes = np.logspace(5, np.floor(np.log2(max_time // 2)), 10, base=2, dtype=int)
    print("Training dataset sizes we will try:", data_sizes.tolist())
    for dataset_size in data_sizes:
        print()
        print("Dataset size", dataset_size)
        # initialize model
        model = model_class(input_size=302).double()
        # train the model on this amount of data
        model, log = optimize_model(
            new_calcium_data,
            model,
            mask,
            num_epochs=num_epochs,
            seq_len=seq_len,
            dataset_size=dataset_size,
        )
        # put the worm and neuron in log
        log["worm"] = worm
        log["neuron_to_idx"] = neuron_to_idx
        log["neuron"] = neuron
        log["num_neurons"] = num_neurons
        # true dataset size
        size = log["dataset_size"]
        # predict with the model
        targets, predictions = model_predict(new_calcium_data, model)
        # log targets and predictions
        log["targets"] = targets
        log["predictions"] = predictions
        if plotting:
            # plot loss curves
            plot_loss_log(
                log,
                plt_title="%s, %s neurons, data size %s, seq. len %s "
                "\n %s Model: Loss curves"
                % (worm.upper(), num_neurons, size, seq_len, model_name),
            )
            # plot prediction for a single neuron
            plot_target_prediction(
                targets[:, neuron_idx],
                predictions[:, neuron_idx],
                plt_title="%s, neuron %s, data size %s, seq. len %s"
                " \n %s Model: Ca2+ residuals prediction"
                % (worm.upper(), neuron, size, seq_len, model_name),
            )
            # plot scatterplot of all predictions
            plot_correlation_scatter(
                targets,
                predictions,
                plt_title="%s, %s neurons,"
                " data size %s, seq. len %s \n %s Model: Correlation of all neuron Ca2+ "
                "residuals" % (worm.upper(), num_neurons, size, seq_len, model_name),
            )
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # add to results
        results_list.append((model, log))
    return results_list


def leave_one_worm_out_training(
    model_class,
    multi_worms_dataset,
    num_epochs=100,
    model_name="",
    seq_len=1,
    plotting=True,
):
    """
    Train on all but one worm in a dataset
    and test on that one worm.
    """
    leaveOut_worm = np.random.choice(list(multi_worms_dataset))
    test_worm_dataset = multi_worms_dataset[leaveOut_worm]
    train_worms_dataset = {
        worm: dataset
        for worm, dataset in multi_worms_dataset.items()
        if worm != leaveOut_worm
    }
    # initialize the model
    model = model_class(input_size=302).double()
    # train the model sequentially on many worms
    train_results = multi_worm_training(
        model_class,
        train_worms_dataset,
        num_epochs,
        model_name,
        seq_len,
        plotting=False,
    )
    model, train_log = train_results[-1]
    # get the calcium data for left out worm
    new_calcium_data = test_worm_dataset["named_data"]
    mask = test_worm_dataset["named_neurons_mask"]
    # get the neuron to idx map
    neuron_to_idx = test_worm_dataset["named_neuron_to_idx"]
    idx_to_neuron = dict((v, k) for k, v in neuron_to_idx.items())
    # pick a neuron
    neuron_idx = np.random.choice(list(neuron_to_idx.keys())) - 1
    neuron = neuron_to_idx[neuron_idx]
    # get max time and number of neurons
    max_time = test_worm_dataset["max_time"]
    num_neurons = test_worm_dataset["num_neurons"]
    # predict with the model
    targets, predictions = model_predict(new_calcium_data, model)
    # create a log for this evaluation
    log = dict()
    log.update(train_log)
    log["worm"] = leaveOut_worm
    log["neuron_to_idx"] = neuron_to_idx
    log["neuron"] = neuron
    log["num_neurons"] = num_neurons
    log["targets"] = targets
    log["predictions"] = predictions
    size = train_log["dataset_size"]
    # test the model on the left out worm
    if plotting:
        # plot final loss curve
        plot_loss_log(
            log,
            plt_title="%s, data size %s, seq. len %s "
            "\n %s Model: Loss curves"
            % (set(w.upper() for w in train_worms_dataset), size, seq_len, model_name),
        )
        # plot prediction for a single neuron
        plot_target_prediction(
            targets[:, neuron_idx],
            predictions[:, neuron_idx],
            plt_title="%s, neuron %s, data size %s, seq. len "
            "%s \n %s Model: Ca2+ residuals prediction"
            % (leaveOut_worm.upper(), neuron, size, seq_len, model_name),
        )
        # plot scatterplot of all neuron predictions
        plot_correlation_scatter(
            targets[:, mask],
            predictions[:, mask],
            plt_title="%s, %s neurons,"
            " data size %s, seq. len %s \n %s Model: Correlation of all neuron Ca2+ "
            "residuals"
            % (leaveOut_worm.upper(), num_neurons, size, seq_len, model_name),
        )
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return log


def multi_worm_training(
    model_class,
    multi_worms_dataset,
    num_epochs=100,
    model_name="",
    seq_len=1,
    plotting=False,
):
    """
    A helper function to investigate the effect of training a model
    on increasingly more worms.
    """
    print("Number of worms in this dataset:", len(multi_worms_dataset))
    results_list = []
    worms_seen = set()
    for worm in multi_worms_dataset:
        print()
        print("Trained so far on", sorted(worms_seen))
        print("Currently training on", worm)
        worms_seen.add(worm)
        # parse worm dataset
        single_worm_dataset = pick_worm(multi_worms_dataset, worm)
        # get the calcium data for this worm
        new_calcium_data = single_worm_dataset["named_data"]
        mask = single_worm_dataset["named_neurons_mask"]
        # get the neuron to idx map
        neuron_to_idx = single_worm_dataset["named_neuron_to_idx"]
        idx_to_neuron = dict((v, k) for k, v in neuron_to_idx.items())
        # pick a neuron
        neuron_idx = np.random.choice(list(neuron_to_idx.keys())) - 1
        neuron = neuron_to_idx[neuron_idx]
        # get max time and number of neurons
        max_time = single_worm_dataset["max_time"]
        num_neurons = single_worm_dataset["num_neurons"]
        # initialize model and an optimizer
        model = model_class(input_size=302).double()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # train the model on this worm's  data
        model, log = optimize_model(
            new_calcium_data,
            model,
            mask,
            optimizer,
            dataset_size=2048,
            num_epochs=num_epochs,
            seq_len=seq_len,
        )
        # put the worm and neuron in log
        log["worm"] = worm
        log["neuron_to_idx"] = neuron_to_idx
        log["neuron"] = neuron
        log["num_neurons"] = num_neurons
        # true dataset size
        size = log["dataset_size"]
        # predict with the model
        targets, predictions = model_predict(new_calcium_data, model)
        # log targets and predictions
        log["targets"] = targets
        log["predictions"] = predictions
        # plot figures
        if plotting:
            # plot loss curves
            plot_loss_log(
                log,
                plt_title="%s, %s neurons, data size %s, seq. len %s "
                "\n %s Model: Loss curves"
                % (worm.upper(), num_neurons, size, seq_len, model_name),
            )
            # plot prediction for a single neuron
            plot_target_prediction(
                targets[:, neuron_idx],
                predictions[:, neuron_idx],
                plt_title="%s, neuron %s, data size %s, seq. len "
                "%s \n %s Model: Ca2+ residuals prediction"
                % (worm.upper(), neuron, size, seq_len, model_name),
            )
            # plot scatterplot of all neuron predictions
            plot_correlation_scatter(
                targets[:, mask],
                predictions[:, mask],
                plt_title="%s, %s neurons,"
                " data size %s, seq. len %s \n %s Model: Correlation of all neuron Ca2+ "
                "residuals" % (worm.upper(), num_neurons, size, seq_len, model_name),
            )
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # add to results
        results_list.append((model, log))
    return results_list


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
        # TODO: figure out if gradient clipping is necessary.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
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
