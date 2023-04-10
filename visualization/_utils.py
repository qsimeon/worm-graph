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
    """
    # process the config.yaml file inside the log folder
    cfg_path = os.path.join(log_dir, "config.yaml")
    if os.path.exists(cfg_path):
        config = OmegaConf.structured(OmegaConf.load(cfg_path))
    else:
        config = OmegaConf.structured(
            {
                "dataset": {"name": "unknown"},
                "model": {"type": "unknown"},
                "timestamp": "unknown",
            }
        )
    # get strings for plot title
    dataset_name = config.dataset.name
    model_name = config.model.type
    timestamp = config.timestamp
    # create the plot title
    plt_title = "Loss curves\nmodel: {}\ndataset: {}\ntime: {}".format(
        model_name,
        dataset_name,
        timestamp,
    )
    # load the loss dataframe
    loss_df = pd.read_csv(os.path.join(log_dir, "loss_curves.csv"), index_col=0)
    # plot loss vs epochs
    plt.figure()
    sns.lineplot(
        x="epochs",
        y="train_losses",
        data=loss_df,
        label="original train",
        color="c",
        alpha=0.8,
        **dict(linestyle=":"),
    )
    sns.lineplot(
        x="epochs",
        y="test_losses",
        data=loss_df,
        label="original test",
        color="r",
        alpha=0.8,
        **dict(linestyle=":"),
    )
    sns.lineplot(x="epochs", y="centered_train_losses", data=loss_df, label="train")
    sns.lineplot(x="epochs", y="centered_test_losses", data=loss_df, label="test")
    plt.legend()
    plt.title(plt_title)
    plt.xlabel("Epoch (# worm cohorts)")
    plt.ylabel("Loss - Baseline")
    plt.savefig(os.path.join(log_dir, "loss_curves.png"))
    plt.close()
    plt.close()
    return None


def plot_before_after_weights(log_dir: str) -> None:
    """
    Plot the model's readout weigths from before and after training.
    """
    # process the config.yaml file inside the log folder
    cfg_path = os.path.join(log_dir, "config.yaml")
    if os.path.exists(cfg_path):
        config = OmegaConf.structured(OmegaConf.load(cfg_path))
    else:
        config = OmegaConf.structured(
            {
                "dataset": {"name": "unknown"},
                "model": {"type": "unknown"},
                "timestamp": "unknown",
            }
        )
    # get strings for plot title
    dataset_name = config.dataset.name
    model_name = config.model.type
    timestamp = config.timestamp
    # create the plot title
    plt_title = "Model readout weights\nmodel: {}\ndataset: {}\ntime: {}".format(
        model_name,
        dataset_name,
        timestamp,
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
    return None


def plot_targets_predictions(
    log_dir: str,
    worm: Union[str, None] = "all",
    neuron: Union[str, None] = "all",
    use_residual: bool = False,
) -> None:
    """
    Plot of the target calcium or calcium residual time series overlayed
    with the predicted calcium or calcium residual time series of a single
    neuron in a given worm.
    """
    # whether using residual or calcium signal
    signal_str = "residual" if use_residual else "calcium"
    # process the config.yaml file inside the log folder
    cfg_path = os.path.join(log_dir, "config.yaml")
    if os.path.exists(cfg_path):
        config = OmegaConf.structured(OmegaConf.load(cfg_path))
    else:
        config = OmegaConf.structured(
            {
                "dataset": {"name": "unknown"},
                "model": {"type": "unknown"},
                "timestamp": "unknown",
            }
        )
    # get strings for plot title
    dataset_name = config.dataset.name
    model_name = config.model.type
    timestamp = config.timestamp
    # recursive call for all worms
    if (worm is None) or (worm.lower() == "all"):
        all_worms = [fname for fname in os.listdir(log_dir) if fname.startswith("worm")]
        for _worm_ in all_worms:
            plot_targets_predictions(log_dir, _worm_, neuron)
        return None
    else:
        assert worm in set(os.listdir(log_dir)), "No data for requested worm found."

    # load predictions dataframe
    predictions_df = pd.read_csv(
        os.path.join(log_dir, worm, "predicted_" + signal_str + ".csv"), index_col=0
    )
    tau = predictions_df["tau"][0]
    # load targets dataframe
    targets_df = pd.read_csv(
        os.path.join(log_dir, worm, "target_" + signal_str + ".csv"), index_col=0
    )

    # plot helper
    def func(_neuron_):
        os.makedirs(os.path.join(log_dir, worm, "figures"), exist_ok=True)
        plt_title = (
            "Neural activity "
            + signal_str
            + " (GCaMP fluorescence) \nworm: {}, neuron: {}\nmodel: {}\ndataset: {}\ntime: {}".format(
                worm,
                _neuron_,
                model_name,
                dataset_name,
                timestamp,
            )
        )
        sns.lineplot(
            data=targets_df,
            x=targets_df.time_in_seconds,
            y=targets_df[_neuron_],
            label="target",
        )
        sns.lineplot(
            data=predictions_df,
            x=targets_df.time_in_seconds,
            y=predictions_df[_neuron_],
            label="predict",
            alpha=0.5,
        )
        ylo, yhi = plt.gca().get_ylim()
        plt.gca().fill_between(
            targets_df.time_in_seconds,
            ylo,
            yhi,
            where=predictions_df["train_test_label"] == "train",
            alpha=0.1,
            facecolor="cyan",
            label="train",
        )
        plt.gca().fill_between(
            targets_df.time_in_seconds,
            ylo,
            yhi,
            where=predictions_df["train_test_label"] == "test",
            alpha=0.1,
            facecolor="magenta",
            label="test",
        )
        plt.gca().fill_between(
            targets_df.time_in_seconds.to_numpy()[-tau:],
            ylo,
            yhi,
            alpha=0.3,
            facecolor="red",
            label="predict",
        )
        plt.legend(loc="upper left", fontsize=6)
        plt.suptitle(plt_title)
        plt.xlabel("Time (seconds)")
        plt.ylabel(signal_str.capitalize() + " ($\Delta F / F$)")
        plt.savefig(
            os.path.join(log_dir, worm, "figures", signal_str + "_%s.png" % _neuron_)
        )
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


def plot_correlation_scatterplot(
    log_dir: str,
    worm: Union[str, None] = "all",
    neuron: Union[str, None] = "all",
    use_residual: bool = False,
):
    """
    Create a scatterpot of the target and predicted calcium or calcium residual
    colored by train and test sample.
    """
    # whether using residual or calcium signal
    signal_str = "residual" if use_residual else "calcium"
    # process the config.yaml file inside the log folder
    cfg_path = os.path.join(log_dir, "config.yaml")
    if os.path.exists(cfg_path):
        config = OmegaConf.structured(OmegaConf.load(cfg_path))
    else:
        config = OmegaConf.structured(
            {
                "dataset": {"name": "unknown"},
                "model": {"type": "unknown"},
                "timestamp": "unknown",
            }
        )
    # get strings for plot title
    dataset_name = config.dataset.name
    model_name = config.model.type
    timestamp = config.timestamp
    # recursive call for all worms
    if (worm is None) or (worm.lower() == "all"):
        all_worms = [fname for fname in os.listdir(log_dir) if fname.startswith("worm")]
        for _worm_ in all_worms:
            plot_correlation_scatterplot(log_dir, _worm_, neuron)
        return None
    else:
        assert worm in set(os.listdir(log_dir)), "No data for requested worm found."
    # load predictions dataframe
    predictions_df = pd.read_csv(
        os.path.join(log_dir, worm, "predicted_" + signal_str + ".csv"), index_col=0
    )
    # load targets dataframe
    targets_df = pd.read_csv(
        os.path.join(log_dir, worm, "target_" + signal_str + ".csv"), index_col=0
    )

    # plot helper
    def func(_neuron_):
        os.makedirs(os.path.join(log_dir, worm, "figures"), exist_ok=True)
        plt_title = "Scatterplot of predicted vs target residuals\nworm: {}, neuron: {}\nmodel: {}\ndataset: {}\ntime: {}".format(
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
        sns.lmplot(
            data=data_df,
            x="target",
            y="prediction",
            hue="label",
            legend=True,
            palette={"test": "magenta", "train": "cyan"},
            scatter_kws={"alpha": 0.1},
            # line_kws={"color": "black"},
        )
        plt.suptitle(plt_title)
        plt.axis("equal")
        plt.gca().set_ylim(plt.gca().get_xlim())
        plt.xlabel("Target " + signal_str + " ($\Delta F / F$)")
        plt.ylabel("Predicted " + signal_str + " ($\Delta F / F$)")
        plt.savefig(
            os.path.join(
                log_dir, worm, "figures", signal_str + "_correlation_%s.png" % _neuron_
            )
        )
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
