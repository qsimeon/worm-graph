from visualization._pkg import *


def draw_connectome(
    network, pos=None, labels=None, plt_title="C. elegans connectome network"
):
    """
    Args:
      network: PyG Data object containing a C. elegans connectome graph.
      pos: dict, mapping of node index to 2D coordinate.
      labels: dict, mapping of node index to neuron name.
    """
    # convert to networkx
    G = torch_geometric.utils.to_networkx(network)
    # create figure
    plt.figure(figsize=(20, 10))
    ## nodes
    inter = [node for i, node in enumerate(G.nodes) if network.y[i] == 0.0]
    motor = [node for i, node in enumerate(G.nodes) if network.y[i] == 1.0]
    other = [node for i, node in enumerate(G.nodes) if network.y[i] == 2.0]
    pharynx = [node for i, node in enumerate(G.nodes) if network.y[i] == 3.0]
    sensory = [node for i, node in enumerate(G.nodes) if network.y[i] == 4.0]
    sexspec = [node for i, node in enumerate(G.nodes) if network.y[i] == 5.0]
    ## edges
    junctions = [
        edge for i, edge in enumerate(G.edges) if network.edge_attr[i, 0] > 0.0
    ]  # gap junctions/electrical synapses encoded as [1,0]
    synapses = [
        edge for i, edge in enumerate(G.edges) if network.edge_attr[i, 1] > 0.0
    ]  # chemical synapse encoded as [0,1]
    ## edge weights
    gap_weights = [int(network.edge_attr[i, 0]) / 50 for i, edge in enumerate(G.edges)]
    chem_weights = [int(network.edge_attr[i, 1]) / 50 for i, edge in enumerate(G.edges)]
    ## metadata
    if pos is None:
        pos = network.pos
    # pos = nx.kamada_kawai_layout(G)
    if labels is None:
        labels = network.id_neuron
    options = {"edgecolors": "tab:gray", "node_size": 500, "alpha": 0.5}
    ## draw nodes
    nx.draw_networkx_edges(
        G, pos, edgelist=junctions, width=gap_weights, alpha=0.5, edge_color="tab:blue"
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=synapses, width=chem_weights, alpha=0.5, edge_color="tab:red"
    )
    nx.draw_networkx_labels(G, pos, labels, font_size=6)
    ## draw edges
    nx.draw_networkx_nodes(G, pos, nodelist=inter, node_color="blue", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=motor, node_color="red", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=other, node_color="green", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=pharynx, node_color="yellow", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=sensory, node_color="magenta", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=sexspec, node_color="cyan", **options)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="inter",
            markerfacecolor="b",
            alpha=0.5,
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="motor",
            markerfacecolor="r",
            alpha=0.5,
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="other",
            markerfacecolor="g",
            alpha=0.5,
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="pharynx",
            markerfacecolor="y",
            alpha=0.5,
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="sensory",
            markerfacecolor="m",
            alpha=0.5,
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="sex",
            markerfacecolor="c",
            alpha=0.5,
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            color="b",
            label="gap junction",
            linewidth=2,
            alpha=0.5,
            markersize=10,
        ),
        Line2D(
            [0], [0], color="r", label="synapse", linewidth=2, alpha=0.5, markersize=10
        ),
    ]
    plt.title(plt_title)
    plt.legend(handles=legend_elements, loc="upper right")
    plt.show()


def plot_loss_curves(log_dir):
    """
    Plot the loss curves stored in the given log directory.
    TODO: Also make a plot of cumulative number of samples vs. epochs.
    """
    # process the log folder name
    timestamp, dataset_name, model_name = str.split(os.path.split(log_dir)[-1], "-")
    plt_title = "Loss curves\nmodel: {}, dataset: {}\ntime: {}".format(
        model_name, dataset_name, timestamp
    )
    # load the loss dataframe
    loss_df = pd.read_csv(os.path.join(log_dir, "loss_curves.csv"), index_col=0)
    plt.figure()
    sns.lineplot(x="epochs", y="centered_train_losses", data=loss_df, label="train")
    sns.lineplot(x="epochs", y="centered_test_losses", data=loss_df, label="test")
    plt.legend()
    plt.title(plt_title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss - Baseline")
    plt.savefig(os.path.join(log_dir, "loss_curves.png"))
    plt.close()
    return None


def plot_before_after_weights(log_dir):
    """
    Plot the model's readout weigths from
    before and after training.
    """
    # process the log folder name
    timestamp, dataset_name, model_name = str.split(os.path.split(log_dir)[-1], "-")
    plt_title = "Model readout weights\nmodel: {}, dataset: {}\ntime: {}".format(
        model_name, dataset_name, timestamp
    )
    # load the first model checkpoint
    chkpt_dir = os.path.join(log_dir, "checkpoints")
    chkpts = sorted(os.listdir(chkpt_dir), key=lambda x: int(x.split("_")[0]))
    first_chkpt = torch.load(os.path.join(chkpt_dir, chkpts[0]))
    last_chkpt = torch.load(os.path.join(chkpt_dir, chkpts[-1]))
    input_size, hidden_size = first_chkpt["input_size"], first_chkpt["hidden_size"]
    model = eval(model_name)(input_size, hidden_size)
    # plot the readout weights
    fig, axs = plt.subplots(1, 2)
    # before training
    model.load_state_dict(first_chkpt["model_state_dict"])
    axs[0].imshow(model.linear.weight.detach().cpu().T)
    axs[0].set_title("Initialized")
    axs[0].set_ylabel("Hidden size")
    axs[0].set_xlabel("Output size")
    # after training
    model.load_state_dict(last_chkpt["model_state_dict"])
    axs[1].imshow(model.linear.weight.detach().cpu().T)
    axs[1].set_title("Trained")
    axs[1].set_xlabel("Output size")
    plt.suptitle(plt_title)
    plt.savefig(os.path.join(log_dir, "readout_weights.png"))
    plt.close()


def plot_targets_predictions(log_dir, worm, neuron="all"):
    """
    Plot of the target Ca2+ residual time series overlayed
    with the predicted Ca2+ residual time series of a single
    neuron in a given worm.
    """
    # process the log folder name
    timestamp, dataset_name, model_name = str.split(os.path.split(log_dir)[-1], "-")

    # load predictions dataframe
    predictions_df = pd.read_csv(
        os.path.join(log_dir, worm, "predicted_ca_residual.csv"), index_col=0
    )
    # load targets dataframe
    targets_df = pd.read_csv(
        os.path.join(log_dir, worm, "target_ca_residual.csv"), index_col=0
    )

    # plot helper
    def func(_neuron_):
        plt_title = "Residual neural activity\nworm: {}, neuron: {}\nmodel: {}, dataset: {}\ntime: {}".format(
            worm,
            _neuron_,
            model_name,
            dataset_name,
            timestamp,
        )
        # TODO: overlay shading on the train and test slices
        sns.lineplot(
            data=targets_df,
            x=targets_df.index,
            y=targets_df[_neuron_],
            label="target",
            # style="train_test_label",
        )
        sns.lineplot(
            data=predictions_df,
            x=predictions_df.index,
            y=predictions_df[_neuron_],
            label="predicted",
            # style="train_test_label",
        )
        plt.legend()
        plt.suptitle(plt_title)
        # plt.axis("square")
        plt.xlabel("Time")
        plt.ylabel("$Ca^{2+}$ Residual ($\Delta F / F$)")
        plt.savefig(os.path.join(log_dir, worm, "%s_residuals.png" % _neuron_))
        plt.close()
        return None

    # plot predictions for neuron(s)
    columns = set(predictions_df.columns)
    if (neuron is None) or (neuron.lower() == "all"):
        for _neuron_ in set(NEURONS_302) & columns:
            func(_neuron_)
    elif neuron in columns:
        func(neuron)
    else:
        pass  # do nothing
    return None


def plot_correlation_scatterplot(log_dir, worm="all", neuron="all"):
    """
    Create a scatterpot of the target and predicted Ca2+
    residuals colored by train and test sample.
    """
    timestamp, dataset_name, model_name = str.split(os.path.split(log_dir)[-1], "-")
    # recursive call for all worms
    if (worm is None) or (worm.lower() == "all"):
        all_worms = [fname for fname in os.listdir(log_dir) if fname.startswith("worm")]
        for _worm_ in all_worms:
            plot_correlation_scatterplot(log_dir, _worm_, neuron)
        return None
    # load predictions dataframe
    predictions_df = pd.read_csv(
        os.path.join(log_dir, worm, "predicted_ca_residual.csv"), index_col=0
    )
    # load targets dataframe
    targets_df = pd.read_csv(
        os.path.join(log_dir, worm, "target_ca_residual.csv"), index_col=0
    )

    # plot helper
    def func(_neuron_):
        plt_title = "Scatterplot of predicted vs target residuals\nworm: {}, neuron: {}\nmodel: {}, dataset: {}\ntime: {}".format(
            worm,
            _neuron_,
            model_name,
            dataset_name,
            timestamp,
        )
        data_dict = {
            "target": targets_df[_neuron_].tolist(),
            "prediction": predictions_df[_neuron_].tolist(),
            "label": predictions_df["train_test_label"].tolist(),
        }
        data_df = pd.DataFrame(data=data_dict)
        sns.lmplot(data=data_df, x="target", y="prediction", hue="label", legend=True)
        plt.suptitle(plt_title)
        plt.axis("equal")
        plt.gca().set_ylim(plt.gca().get_xlim())
        plt.xlabel("Target residual $\Delta F / F$")
        plt.ylabel("Predicted residual $\Delta F / F$")
        plt.savefig(os.path.join(log_dir, worm, "%s_correlation.png" % _neuron_))
        plt.close()
        return None

    # plot predictions for neuron(s)
    columns = set(predictions_df.columns)
    if (neuron is None) or (neuron.lower() == "all"):
        for _neuron_ in set(NEURONS_302) & columns:
            func(_neuron_)
    elif neuron in columns:
        func(neuron)
    else:
        pass  # do nothing
    return None


def plot_hidden_experiment(hidden_experiment):
    """
    Plot the results from the logs returned in `hidden_experiment`
    by `lstm_hidden_size_experiment`.
    """
    color = plt.cm.YlGnBu(np.linspace(0, 1, len(hidden_experiment) + 2))
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=color[2:])
    fig, axs = plt.subplots(1, 2)

    for hs in sorted(hidden_experiment):
        axs[0].plot(
            hidden_experiment[hs]["epochs"],
            np.log10(hidden_experiment[hs]["train_losses"]),
            label="hidden_size=%d" % hs,
            linewidth=2,
        )

    axs[0].plot(
        hidden_experiment[hs]["epochs"],
        np.log10(hidden_experiment[hs]["base_train_losses"]),
        linewidth=2,
        color="r",
        label="baseline",
    )
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("log MSE")
    axs[0].set_title("Training")

    plt.gca().set_prop_cycle(None)
    for hs in sorted(hidden_experiment):
        axs[1].plot(
            hidden_experiment[hs]["epochs"],
            np.log10(hidden_experiment[hs]["test_losses"]),
            label="hidden_size=%d" % hs,
            linewidth=2,
        )

    axs[1].plot(
        hidden_experiment[hs]["epochs"],
        np.log10(hidden_experiment[hs]["base_test_losses"]),
        color="r",
        label="baseline",
    )
    axs[1].set_xlabel("Epoch")
    axs[1].set_yticks(axs[0].get_yticks())
    axs[1].legend(loc="upper right", borderpad=0, labelspacing=0)
    axs[1].set_title("Validation")
    fig.suptitle("LSTM network model loss curves with various hidden units")
    plt.show()


def plot_more_data_losses(results, plt_title=""):
    """
    Makes a plot of the aggregated loss curves on multiple dataset
    sizes from the `results` returned by `more_data_training`.
    """
    num_datasets = len(results)
    fig, ax = plt.subplots(1, 1)

    cmap = plt.get_cmap("PuRd", num_datasets + 2)
    for i, res in enumerate(results):
        label = None if i < num_datasets - 1 else "train"
        model, log = res
        ax.plot(
            log["epochs"],
            np.log10(log["train_losses"]),
            color=cmap(i),
            linewidth=2,
            label=label,
        )

    cmap = plt.get_cmap("YlGnBu", num_datasets + 2)
    for i, res in enumerate(results):
        label = None if i < num_datasets - 1 else "test"
        model, log = res
        ax.plot(
            log["epochs"],
            np.log10(log["test_losses"]),
            color=cmap(i),
            linewidth=2,
            label=label,
        )

    avg_base_train_loss = 0
    for i, res in enumerate(results):
        model, log = res
        avg_base_train_loss += np.log10(log["base_train_losses"])
    avg_base_train_loss /= i + 1
    ax.plot(
        log["epochs"],
        avg_base_train_loss,
        color="purple",
        linewidth=4,
        linestyle="--",
        alpha=0.6,
        label="avg. train baseline",
    )

    avg_base_test_loss = 0
    for i, res in enumerate(results):
        model, log = res
        avg_base_test_loss += np.log10(log["base_test_losses"])
    avg_base_test_loss /= i + 1
    ax.plot(
        log["epochs"],
        avg_base_test_loss,
        color="green",
        linewidth=3,
        linestyle="-.",
        alpha=1,
        label="avg. test baseline",
    )

    ax.legend(title="%s dataset sizes" % num_datasets, labelspacing=0, loc="lower left")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log MSE")
    ax.set_title(plt_title)
    plt.show()


def plot_multi_worm_losses(results, plt_title=""):
    """
    Makes a plot of the aggregated loss curves on multiple worms
    from the `results` returned by `multi_worm_training`.
    """
    num_worms = len(results)
    fig, ax = plt.subplots(1, 1)

    cmap = plt.get_cmap("Reds", num_worms + 2)
    for i, res in enumerate(results):
        label = None if i < num_worms - 1 else "train"
        model, log = res
        ax.plot(
            log["epochs"],
            np.log10(log["train_losses"]),
            color=cmap(i + 2),
            linewidth=2,
            label=label,
        )

    cmap = plt.get_cmap("Blues", num_worms + 2)
    for i, res in enumerate(results):
        label = None if i < num_worms - 1 else "test"
        model, log = res
        ax.plot(
            log["epochs"],
            np.log10(log["test_losses"]),
            color=cmap(i + 2),
            linewidth=2,
            label=label,
        )

    avg_base_train_loss = 0
    for i, res in enumerate(results):
        model, log = res
        avg_base_train_loss += np.log10(log["base_train_losses"])
    avg_base_train_loss /= i + 1
    ax.plot(
        log["epochs"],
        avg_base_train_loss,
        color="r",
        linewidth=4,
        linestyle="--",
        alpha=0.6,
        label="avg. train baseline",
    )

    avg_base_test_loss = 0
    for i, res in enumerate(results):
        model, log = res
        avg_base_test_loss += np.log10(log["base_test_losses"])
    avg_base_test_loss /= i + 1
    ax.plot(
        log["epochs"],
        avg_base_test_loss,
        color="b",
        linewidth=3,
        linestyle="-.",
        alpha=1,
        label="avg. test baseline",
    )

    ax.legend(title="%s worms" % num_worms, labelspacing=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log MSE")
    ax.set_title(plt_title)
    plt.show()


def plot_neuron_train_test_samples(
    single_worm_dataset, neuron_idx, num_samples, seq_len, tau
):
    """
    Visualizes example train and test samples for a single neuron from the worm.
    Args:
      single_worm_dataset: dict, dataset for a single worm.
      neuron_idx: int, index in the neuron in the dataset.
      num_samples: int, the number of train (or test) examples to plot.
      seq_len: int, the length of the input (or target) time series.
      tau: int, the amount the target series is shifted by.
    """
    neuron_to_idx = single_worm_dataset["named_neuron_to_idx"]
    calcium_data = single_worm_dataset["named_data"]
    if len(neuron_to_idx) == 0:
        neuron_to_idx = single_worm_dataset["all_neuron_to_idx"]
        calcium_data = single_worm_dataset["all_data"]
    idx_to_neuron = dict((v, k) for k, v in neuron_to_idx.items())
    max_time = single_worm_dataset["max_time"]
    neuron = idx_to_neuron[neuron_idx]
    n_ex = num_samples
    yshifts = np.random.uniform(low=0.5, high=1.0, size=n_ex)
    eps = 0.05
    # datasets (only for visualizing)
    train_dataset = NeuralActivityDataset(
        calcium_data[: max_time // 2],
        tau=tau,
        seq_len=seq_len,
        size=n_ex,
        increasing=True,
        reverse=True,
    )
    test_dataset = NeuralActivityDataset(
        calcium_data[max_time // 2 :],
        tau=tau,
        seq_len=seq_len,
        size=n_ex,
        increasing=False,
        reverse=False,
    )
    # dataloaders (only for visualizing)
    train_sampler = BatchSampler(train_dataset.batch_indices)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_sampler
    )
    test_sampler = BatchSampler(test_dataset.batch_indices)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_sampler)
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    plt.gca().set_prop_cycle(None)
    color = plt.cm.Paired(np.linspace(0, 1, 12))
    axs[0].set_prop_cycle(mpl.cycler(color=color))
    # training set
    trainX, trainY, metadata = next(iter(train_loader))
    batch_indices = np.random.choice(
        trainX.shape[0], n_ex, replace=False
    )  # pick n examples
    for _, batch_idx in enumerate(batch_indices):
        yshift = yshifts[_]
        # input sequence
        axs[0].plot(
            range(metadata["start"][batch_idx], metadata["end"][batch_idx]),
            yshift + trainX[batch_idx, :, neuron_idx].cpu(),
            linewidth=2,
        )
        # target sequence
        axs[0].plot(
            range(
                metadata["start"][batch_idx] + metadata["tau"][batch_idx],
                metadata["end"][batch_idx] + metadata["tau"][batch_idx],
            ),
            eps + yshift + trainY[batch_idx, :, neuron_idx].cpu(),
            linewidth=1,
        )
    axs[0].axvline(x=train_dataset.max_time, c="k", linestyle="--")
    axs[0].set_xlabel("time (s)")
    axs[0].set_yticks([])
    axs[0].set_ylabel("$\Delta F/F$ (random offsets)")
    axs[0].set_title("train set")
    plt.gca().set_prop_cycle(None)
    color = plt.cm.Paired(np.linspace(0, 1, 12))
    axs[1].set_prop_cycle(mpl.cycler(color=color))
    # validation/test set
    testX, testY, metadata = next(iter(test_loader))
    batch_indices = np.random.choice(
        testX.shape[0], n_ex, replace=False
    )  # pick n examples
    for _, batch_idx in enumerate(batch_indices):
        yshift = yshifts[_]
        axs[1].plot(
            range(
                train_dataset.max_time + metadata["start"][batch_idx],
                train_dataset.max_time + metadata["end"][batch_idx],
            ),
            yshift + testX[batch_idx, :, neuron_idx].cpu(),
            linewidth=2,
        )
        axs[1].plot(
            range(
                train_dataset.max_time
                + metadata["start"][batch_idx]
                + metadata["tau"][batch_idx],
                train_dataset.max_time
                + metadata["end"][batch_idx]
                + metadata["tau"][batch_idx],
            ),
            eps + yshift + testY[batch_idx, :, neuron_idx].cpu(),
            linewidth=1,
        )
    axs[1].axvline(x=train_dataset.max_time, c="k", linestyle="--")
    axs[1].set_xlabel("time (s)")
    axs[1].set_yticks([])
    axs[1].set_ylabel("$\Delta F/F$ (random offsets)")
    axs[1].set_title("test set")
    miny = min(axs[0].get_ylim()[0], axs[1].get_ylim()[0])
    maxy = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
    axs[0].set_ylim([miny, maxy])
    axs[1].set_ylim([miny, maxy])
    plt.suptitle(
        "%s Example Samples of the Ca2+ Signal "
        "from Neuron %s, $L$ = %s, τ = %s" % (n_ex, neuron, seq_len, tau)
    )
    plt.show()


def plot_single_neuron_signals(log_dir, worm, neuron, n_lags=20):
    """
    Visualizes the Ca2+ neural activity time series, the first difference
    (i.e. residual) of the former, and the partial autocorrelation function
    of the specified neuron in the given worm.
    """
    # process the log folder name
    dataset_name, model_name, timestamp = str.split(os.path.split(log_dir)[-1], "-")
    plt_title = "Neural activity and partial autocorrelation\nworm: {}, neuron: {}\nmodel: {}, dataset: {}\ntime: {}".format(
        worm, neuron, model_name, dataset_name, timestamp
    )
    # load calcium activity dataframe
    calcium_df = pd.read_csv(
        os.path.join(log_dir, worm, "ca_activity.csv"), index_col=0
    )
    # plot the full Ca2+ recording
    plt.figure()
    fig, axs = plt.subplots(1, 2)
    sns.lineplot(ax=axs[0], data=calcium_df, x=calcium_df.index, y=calcium_df[neuron])
    axs[0].set_title("Neural activty as measured by change in Ca2+ signal")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("GCaMP fluorescence ($\Delta F / F$)")
    # plot the autocorrelation function
    tsaplots.plot_pacf(calcium_df[neuron], ax=axs[1], lags=n_lags)
    axs[1].set_title("Partial autocorrelation function (first %s lags)" % n_lags)
    axs[1].set_xlabel("Lag ($\{tau}$)")
    axs[1].set_ylabel("correlation coefficient")
    plt.suptitle(plt_title)
    plt.show()
    return None


def plot_worm_data(single_worm_dataset, worm_name):
    """
    Plot the neural activity traces for some neurons in a given worm.
    single_worm_dataset: dict, the data for this worm.
    worm_name: str, name to give the worm.
    """
    # get the calcium data and neuron labels
    neuron_to_idx = single_worm_dataset["named_neuron_to_idx"]
    calcium_data = single_worm_dataset["named_data"]
    if len(neuron_to_idx) == 0:
        neuron_to_idx = single_worm_dataset["all_neuron_to_idx"]
        calcium_data = single_worm_dataset["all_data"]
    idx_to_neuron = dict((v, k) for k, v in neuron_to_idx.items())
    # filter for named neurons
    named_neurons = [key for key, val in idx_to_neuron.items() if not val.isnumeric()]
    if not named_neurons:
        named_neurons = list(idx_to_neuron.keys())
    # randomly select 10 neurons to plot
    inds = np.random.choice(named_neurons, 10)
    labels = [idx_to_neuron[i] for i in named_neurons]
    # plot calcium activity traces
    color = plt.cm.tab20(range(10))
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=color)
    plt.figure()
    plt.plot(calcium_data[:200, inds, 0])  # Ca traces in 1st dim
    plt.legend(
        labels,
        title="neuron ID",
        loc="upper right",
        fontsize="x-small",
        title_fontsize="small",
    )
    plt.xlabel("time")
    plt.ylabel("$\delta F / F$")
    plt.title(
        "{}: Calcium traces (first 200 timesteps) of 10 neurons".format(worm_name)
    )
    plt.show()
    return None
