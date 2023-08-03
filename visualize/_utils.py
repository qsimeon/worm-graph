from visualize._pkg import *


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
        labels = network.idx_to_neuron
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


def plot_frequency_distribution(data, ax, title, dt=0.5):
    """Plots the frequency distribution of a signal.

    Args:
        data (list): The signal data.
        ax (matplotlib.axes.Axes): The axes to plot on.
        title (str): The title of the plot.
        dt (float, optional): The time step between samples. Defaults to 0.5.
    """
    # Compute the FFT and frequencies
    fft_data = torch.fft.rfft(torch.tensor(data))
    freqs = torch.fft.rfftfreq(len(data), d=dt)

    # Plot the frequency distribution
    ax.plot(freqs, torch.abs(fft_data))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)


def plot_loss_curves(log_dir):
    """Plots the loss curves stored in a log directory.

    Args:
        log_dir (str): The path to the log directory.
    """

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.set(style='whitegrid')
    sns.set_palette("tab10")

    # process the pipeline_info.yaml file inside the log folder
    cfg_path = os.path.join(log_dir, "pipeline_info.yaml")
    if os.path.exists(cfg_path):
        config = OmegaConf.structured(OmegaConf.load(cfg_path))
        config = config.submodule
    else:
        raise FileNotFoundError("No pipeline_info.yaml file found in {}".format(log_dir))

    # Get strings for plot title
    dataset_name = config.dataset.train.name
    dataset_name = dataset_name.split("_")
    dataset_name = [ds_name[:-4] for ds_name in dataset_name]
    dataset_name = ", ".join(dataset_name) # Just initials
    num_worms = config.dataset.train.num_worms
    model_name = config.model.type
    tau_in = config.train.tau_in

    # Create the plot title
    plt_title = (
        "Model: {}\nNb. of worms: {}\nDataset(s): {}\nTraining {}: {}".format(
            model_name,
            num_worms,
            dataset_name,
            r'$\tau$',
            tau_in,
        )
    )

    # return if no loss curves file found
    loss_curves_csv = os.path.join(log_dir, "loss_curves.csv")
    if not os.path.exists(loss_curves_csv):
        print("No loss curves found in log directory.")
        return None
    
    # load the loss dataframe
    loss_df = pd.read_csv(loss_curves_csv, index_col=0)

    sns.lineplot(
        x="epochs",
        y="base_train_losses",
        data=loss_df,
        ax=ax,
        label="Train baseline",
        color="c",
        alpha=0.8,
        **dict(linestyle=":"),
    )

    sns.lineplot(
        x="epochs",
        y="base_test_losses",
        data=loss_df,
        ax=ax,
        label="Test baseline",
        color="r",
        alpha=0.8,
        **dict(linestyle=":"),
    )
    sns.lineplot(x="epochs", y="train_losses", data=loss_df, ax=ax, label="Train")
    sns.lineplot(x="epochs", y="test_losses", data=loss_df, ax=ax, label="Test")
    plt.legend(frameon=True, loc="upper right", fontsize=12)

    plt.title('Loss curves', fontsize=16)
    
    x_position_percent = 0.075  # Adjust this value to set the desired position
    x_position_box = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * x_position_percent
    y_position_percent = 0.80  # Adjust this value to set the desired position
    y_position_box = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * y_position_percent
    plt.text(x_position_box, y_position_box, plt_title, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), style='italic')
    plt.grid(True)
    plt.xlabel("Epoch (# worm cohorts)")
    plt.ylabel("Loss")
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(log_dir, "loss_curves.png"))
    plt.close()

    return None


def plot_before_after_weights(log_dir: str) -> None:
    """Plots the model's readout weights from before and after training.

    Args:
        log_dir (str): The path to the log directory.

    Returns:
        None
    """
    # process the pipeline_info.yaml file inside the log folder
    cfg_path = os.path.join(log_dir, "pipeline_info.yaml")
    if os.path.exists(cfg_path):
        config = OmegaConf.structured(OmegaConf.load(cfg_path))
        config = config.submodule
    else:
        raise FileNotFoundError("No pipeline_info.yaml file found in {}".format(log_dir))
    
    # get strings for plot title
    dataset_name = config.dataset.train.name
    model_name = config.model.type
    tau_in = config.train.tau_in
    # Create the plot title
    plt_title = "Model readout weights\nmodel: {}\ndataset: {}\ntraining tau: {}".format(
        model_name,
        dataset_name,
        tau_in,
    )
    # return if no checkpoints found
    chkpt_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(chkpt_dir):
        print("No checkpoints found in log directory.")
        return None
    # load the first model checkpoint
    chkpts = sorted(os.listdir(chkpt_dir), key=lambda x: int(x.split("_")[0]))
    checkpoint_zero_dir = os.path.join(chkpt_dir, chkpts[0])
    checkpoint_last_dir = os.path.join(chkpt_dir, chkpts[-1])
    first_chkpt = torch.load(checkpoint_zero_dir, map_location=DEVICE)
    last_chkpt = torch.load(checkpoint_last_dir, map_location=DEVICE)
    # create the model
    input_size, hidden_size, num_layers = (
        first_chkpt["input_size"],
        first_chkpt["hidden_size"],
        first_chkpt["num_layers"],
    )
    loss_name = first_chkpt["loss_name"]
    fft_reg_param, l1_reg_param = (
        first_chkpt["fft_reg_param"],
        first_chkpt["l1_reg_param"],
    )
    model = eval(model_name)(
        input_size,
        hidden_size,
        num_layers,
        loss=loss_name,
        fft_reg_param=fft_reg_param,
        l1_reg_param=l1_reg_param,
    )
    # plot the readout weights
    # Create a figure with a larger vertical size
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    # before training
    model.load_state_dict(first_chkpt["model_state_dict"])
    model.eval()
    weights_before = copy.deepcopy(model.linear.weight.detach().cpu().T)

    # after training
    model.load_state_dict(last_chkpt["model_state_dict"])
    model.eval()
    weights_after = copy.deepcopy(model.linear.weight.detach().cpu().T)

    # check if the weights changed very much
    if torch.allclose(weights_before, weights_after):
        print("Model weights did not change much during training.")
    # print what percentage of weights are close
    print(
        "Model weights changed by {:.2f}% during training.".format(
            100
            * (
                1
                - torch.isclose(weights_before, weights_after, atol=1e-8).sum().item()
                / weights_before.numel()
            )
        ),
        end="\n\n",
    )

    # find min and max across both datasets for the colormap
    vmin = min(weights_before.min(), weights_after.min())
    vmax = max(weights_before.max(), weights_after.max())

    # plot
    im1 = axs[0].imshow(weights_before, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axs[0].set_title("Initialized")
    axs[0].set_ylabel("Hidden size")
    axs[0].set_xlabel("Output size")

    im2 = axs[1].imshow(weights_after, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axs[1].set_title("Trained")
    axs[1].set_xlabel("Output size")

    # create an axes on the right side of axs[1]. The width of cax will be 5%
    # of axs[1] and the padding between cax and axs[1] will be fixed at 0.05 inch.
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax)

    # After creating your subplots and setting their titles, adjust the plot layout
    plt.tight_layout(
        rect=[0, 0, 1, 0.92]
    )  # Adjust the rectangle's top value as needed to give the suptitle more space
    plt.subplots_adjust(top=0.85)  # Adjust the top to make space for the title
    # Now add your suptitle, using the y parameter to control its vertical placement
    plt.suptitle(
        plt_title, fontsize="medium", y=0.95
    )  # Adjust y as needed so the title doesn't overlap with the plot
    # Save and close as before
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
    # Whether using residual or calcium signal
    signal_str = "residual" if use_residual else "calcium"

    # Process the pipeline_info.yaml file inside the log folder
    cfg_path = os.path.join(log_dir, "pipeline_info.yaml")
    if os.path.exists(cfg_path):
        config = OmegaConf.structured(OmegaConf.load(cfg_path))
        config = config.submodule
    else:
        raise FileNotFoundError(
            f"Could not find pipeline_info.yaml file in {log_dir}."
        )

    # Get strings for plot title
    predict_dataset_name = config.dataset.predict.name
    predict_dataset_name = predict_dataset_name.split("_")
    predict_dataset_name = [ds_name[:-4] for ds_name in predict_dataset_name]
    predict_dataset_name = ", ".join(predict_dataset_name)

    model_name = config.model.type
    tau_out = config.predict.tau_out
    
    # Recursive call for all worms
    if (worm is None) or (worm.lower() == "all"):
        all_worms = [fname for fname in os.listdir(log_dir) if fname.startswith("worm")]
        for _worm_ in all_worms:
            plot_targets_predictions(log_dir, _worm_, neuron)
        return None
    else:
        assert worm in set(os.listdir(log_dir)), "No data for requested worm found."

    # Return if no targets or predicitions files found
    predictions_csv = os.path.join(log_dir, worm, "predicted_" + signal_str + ".csv")
    targets_csv = os.path.join(log_dir, worm, "target_" + signal_str + ".csv")
    if (not os.path.exists(predictions_csv)) or (not os.path.exists(targets_csv)):
        print("No targets or predictions found in log directory.")
        return None
    # Load predictions dataframe
    predictions_df = pd.read_csv(predictions_csv, index_col=0)
    tau_out = predictions_df["tau"][0]
    # Load targets dataframe
    targets_df = pd.read_csv(targets_csv, index_col=0)

    # Plot helper
    def func(_neuron_):
        os.makedirs(os.path.join(log_dir, worm, "figures"), exist_ok=True)
        
        # Create the plot title
        plt_title = (
            "Model: {}\nPredict dataset: {}\nWorm index: {}\nPrediction {}: {}".format(
                model_name,
                predict_dataset_name,
                worm,
                r'$\tau$',
                tau_out,
            )
        )

        # Create a figure with a larger size. Adjust (8, 6) as per your need.
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Use sns whitegrid style
        sns.set_style("whitegrid")
        # Use palette tab10
        sns.set_palette("tab10")

        sns.lineplot(
            data=targets_df,
            x=targets_df.time_in_seconds,
            y=targets_df[_neuron_],
            label="Target",
            color=sns.color_palette("tab10")[3], #red
            linestyle="dashed",
            alpha=0.5,
            linewidth=1.5,
            ax=ax,
        )

        sns.lineplot(
            data=predictions_df,
            x=targets_df.time_in_seconds,
            y=predictions_df[_neuron_],
            color=sns.color_palette("tab10")[0], #blue
            label="Prediction",
            alpha=1.0,
            linewidth=1.1,
            ax=ax,
        )

        ylo, yhi = ax.get_ylim()

        ax.fill_between(
            targets_df.time_in_seconds,
            ylo,
            yhi,
            where=predictions_df["train_test_label"] == "train",
            alpha=0.2,
            facecolor=sns.color_palette("hls", 8)[0], #red
            label="Train stage",
        )

        ax.fill_between(
            targets_df.time_in_seconds,
            ylo,
            yhi,
            where=predictions_df["train_test_label"] == "test",
            alpha=0.2,
            facecolor=sns.color_palette("hls", 8)[1], #yellow
            label="Test stage",
        )

        ax.fill_between(
            targets_df.time_in_seconds.to_numpy()[-tau_out:],
            ylo,
            yhi,
            alpha=0.2,
            facecolor=sns.color_palette("hls", 8)[2], #green
            label="Prediction stage",
        )

        ax.legend(loc="upper left", fontsize='small')
        
        # Adjust the plot layout
        x_position_percent = 0.01  # Adjust this value to set the desired position
        x_position_box = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * x_position_percent
        y_position_box = min(targets_df[_neuron_].min(), predictions_df[_neuron_].min())
        plt.text(x_position_box, y_position_box, plt_title, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), style='italic')
        plt.title("Neural activity: {} (GCaMP fluorescence) - {}".format(signal_str, _neuron_))
        plt.xlabel("Time (s)")
        plt.ylabel(signal_str.capitalize() + " ($\Delta F / F$)")
        plt.tight_layout()
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
    # Whether using residual or calcium signal
    signal_str = "residual" if use_residual else "calcium"

    # Process the pipeline_info.yaml file inside the log folder
    cfg_path = os.path.join(log_dir, "pipeline_info.yaml")
    if os.path.exists(cfg_path):
        config = OmegaConf.structured(OmegaConf.load(cfg_path))
        config = config.submodule
    else:
        config = OmegaConf.structured(
            {
                "dataset": {"name": "unknown"},
                "model": {"type": "unknown"},
                "train": {"tau_in": "unknown"},
                "predict": {"tau_out": "unknown", "dataset": {"name": "unknown"}},
                "globals": {"timestamp": datetime.now().strftime("%Y_%m_%d_%H_%M_%S")},
            }
        )

    # Get strings for plot title
    predict_dataset_name = config.dataset.predict.name
    predict_dataset_name = predict_dataset_name.split("_")
    predict_dataset_name = [ds_name[:-4] for ds_name in predict_dataset_name]
    predict_dataset_name = ", ".join(predict_dataset_name)
    
    model_name = config.model.type
    tau_out = config.predict.tau_out

    # Recursive call for all worms
    if (worm is None) or (worm.lower() == "all"):
        all_worms = [fname for fname in os.listdir(log_dir) if fname.startswith("worm")]
        for _worm_ in all_worms:
            plot_correlation_scatterplot(log_dir, _worm_, neuron)
        return None
    else:
        assert worm in set(os.listdir(log_dir)), "No data for requested worm found."

    # Load predictions dataframe
    predictions_df = pd.read_csv(
        os.path.join(log_dir, worm, "predicted_" + signal_str + ".csv"), index_col=0
    )
    tau_out = predictions_df["tau"][0]
    # Load targets dataframe
    targets_df = pd.read_csv(
        os.path.join(log_dir, worm, "target_" + signal_str + ".csv"), index_col=0
    )

    # TODO: consider only the predictions and targets for the last tau_out indices
    predictions_df = predictions_df.iloc[-tau_out:, :]
    targets_df = targets_df.iloc[-tau_out:, :]

    # Plot helper
    def func(_neuron_):
        os.makedirs(os.path.join(log_dir, worm, "figures"), exist_ok=True)

        # Create a figure with a larger size
        fig, ax = plt.subplots(figsize=(8, 5))

        # Use sns whitegrid style
        sns.set_style("whitegrid")
        # Use palette tab10
        sns.set_palette("tab10")

        data_dict = {
            "target": targets_df[_neuron_].tolist(),
            "prediction": predictions_df[_neuron_].tolist(),
            "label": predictions_df["train_test_label"].tolist(),
        }

        data_df = pd.DataFrame(data=data_dict)

        # Create scatterplot of predicted vs target
        sns.scatterplot(
            data=data_df,
            x="target",
            y="prediction",
            hue="label",
            legend=False,
            ax=ax,
            size=0.5,
        )

        # Linear regression betwee target and prediction
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            data_df["target"], data_df["prediction"]
        )

        # Create label for linear regression line (curve + R2)
        linreg_label = "y = {:.2f}x + {:.2f}".format(
            slope, intercept
        )

        # Add linear regression line
        sns.lineplot(
            x=data_df["target"],
            y=intercept + slope * data_df["target"],
            color="black",
            legend=False,
            ax=ax,
        )
        
        # Create the plot textbox
        plt_title = (
            "Model: {}\nPredict dataset: {}\nWorm index: {}\nPrediction {}: {}\n\n{}\n$R^2$: {}".format(
                model_name,
                predict_dataset_name,
                worm,
                r'$\tau$',
                tau_out,
                #timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                linreg_label,
                round(r_value ** 2, 4),
            )
        )

        # Adjust x box position
        x_position_percent = 0.02  # Adjust this value to set the desired position
        x_position_box = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * x_position_percent

        # Adjust y box position
        y_position_percent = 0.03  # Adjust this value to set the desired position
        y_position_box = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * y_position_percent

        plt.text(x_position_box, y_position_box, plt_title, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', alpha=0.5), style='italic')

        plt.xlabel("Target " + signal_str + " ($\Delta F / F$)")
        plt.ylabel("Predicted " + signal_str + " ($\Delta F / F$)")

        plt.title("{} scatter plot: predicted vs. target values - {}".format(signal_str.title(), _neuron_))

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                log_dir, worm, "figures", signal_str + "_correlation_%s.png" % _neuron_
            )
        )
        plt.close()
        return None

    # Plot predictions for neuron(s)
    columns = set(predictions_df.columns)
    if (neuron is None) or (neuron.lower() == "all"):
        for _neuron_ in set(NEURONS_302) & columns:
            func(_neuron_)
    elif neuron in columns:
        func(neuron)
    else:
        pass  # do nothing
    return None


def plot_worm_data(worm_data, num_neurons=5, smooth=False):
    """
    Plot a few calcium traces from a given worm's data.

    :param worm_data: The data for a single worm.
    :param num_neurons: The number of neurons to plot.
    """

    np.random.seed(42) # set random seed for reproducibility

    worm = worm_data["worm"]
    dataset = worm_data["dataset"]
    if smooth:
        calcium_data = worm_data["smooth_calcium_data"]
    else:
        calcium_data = worm_data["calcium_data"]
    time_in_seconds = worm_data["time_in_seconds"]
    slot_to_named_neuron = worm_data["slot_to_named_neuron"]
    neuron_indices = set(
        np.random.choice(list(slot_to_named_neuron.keys()), num_neurons, replace=True)
    )

    for neuron_idx in neuron_indices:
        neuron_name = slot_to_named_neuron.get(neuron_idx, None)
        if neuron_name is not None:
            plt.plot(time_in_seconds, calcium_data[:, neuron_idx], label=neuron_name)
        else:
            ValueError("No neurons with data were selected.")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Calcium Signal")
    plt.title(
        f"Dataset: {dataset}, Worm: {worm}\nCalcium Traces of Random {num_neurons} Neurons"
    )
    plt.legend()
    plt.show()


def plot_heat_map(
    matrix,
    title=None,
    cmap=None,
    xlabel=None,
    ylabel=None,
    xticks=None,
    yticks=None,
    show_xticks=True,
    show_yticks=True,
    xtick_skip=None,
    ytick_skip=None,
    center=None,
    vmin=None,
    vmax=None,
    mask=None,
):
    # Generate the mask
    if type(mask) is str:
        if mask.upper() == "UPPER_T":
            mask = np.triu(np.ones_like(matrix))
        elif mask.upper() == "LOWER_T":
            mask = np.tril(np.ones_like(matrix))

    # Set the figure size
    fig, ax = plt.subplots(figsize=(10, 8))

    # Generate the heatmap
    ax = sns.heatmap(matrix, mask=mask, cmap=cmap, center=center, vmin=vmin, vmax=vmax)

    # Set the title if provided
    if title is not None:
        ax.set_title(title)

    # Set the x and y ticks if provided and show_ticks is True
    if show_xticks:
        if xticks is not None:
            ax.set_xticks(np.arange(len(xticks))[::xtick_skip])
            ax.set_xticklabels(xticks[::xtick_skip], fontsize="small", rotation=90)
    else:
        ax.set_xticks([])

    if show_yticks:
        if yticks is not None:
            ax.set_yticks(np.arange(len(yticks))[::ytick_skip])
            ax.set_yticklabels(yticks[::ytick_skip], fontsize="small", rotation=0)
    else:
        ax.set_yticks([])

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Show the plot
    plt.tight_layout()

def seconds_per_epoch_plot(exp_log_dir, key, log_scale=True):

    if key == 'num_worms':
        key_name = 'Number of worms'
    elif key == 'num_named_neurons':
        key_name = 'Number of named neurons'
    elif key == 'loss':
        key_name = 'Loss'
    elif key == 'seq_len':
        key_name = 'Sequence length'
    elif key == 'num_samples':
        key_name = 'Number of samples'
    elif key == 'hidden_size':
        key_name = 'Hidden size'
    elif key == 'worm_timesteps':
        key_name = 'Worm timesteps'
    elif key == 'optimizer':
        key_name = 'Optimizer'
    elif key == 'learn_rate':
        key_name = 'Learning rate'
    else:
        print('Experiment not registered.\n Registered experiments: \
              num_worms, num_named_neurons, loss, seq_len, num_samples, hidden_size, worm_timesteps, optimizer or learn_rate')
        # End execution
        return None

    # Store seconds per epoch, number of named neurons and number of worms
    seconds_per_epoch = []
    num_named_neurons = []
    num_worms = []
    loss_type = []
    seq_len = []
    num_samples = []
    hidden_size = []
    worm_timesteps = []
    optimizer = []
    lr = []

    # Loop over all folders inside the log directory
    for folder in os.listdir(exp_log_dir):
        # Skip .submitit folder and multirun.yaml file
        if folder == '.submitit' or folder == 'multirun.yaml' or folder == 'scaling_laws':
            continue
        # Load the config file inside each folder
        pipeline_info = OmegaConf.load(os.path.join(exp_log_dir, folder, 'pipeline_info.yaml')).submodule
        train_info = OmegaConf.load(os.path.join(exp_log_dir, folder, 'train_info.yaml'))
        # Extract seconds_per_epoch from the config file
        seconds_per_epoch.append(train_info['seconds_per_epoch'])
        # Extract num_named_neurons from the config file
        num_named_neurons.append(pipeline_info['dataset']['train']['num_named_neurons'])
        # Extract num_worms from the config file
        num_worms.append(pipeline_info['dataset']['train']['num_worms'])
        # Extract loss_type from the config file
        loss_type.append(pipeline_info['model']['loss'])
        # Extract seq_len from the config file
        seq_len.append(pipeline_info['train']['seq_len'])
        # Extract num_samples from the config file
        num_samples.append(pipeline_info['train']['num_samples'])
        # Extract hidden_size from the config file
        hidden_size.append(pipeline_info['model']['hidden_size'])
        # Extract worm_timesteps from the config file
        worm_timesteps.append(train_info['worm_timesteps'])
        # Extract optimizer
        optimizer.append(pipeline_info['train']['optimizer'])
        # Extract learn_rate
        lr.append(pipeline_info['train']['learn_rate'])

    # Create a dataframe with the seconds per epoch, number of named neurons and number of worms
    df = pd.DataFrame({'seconds_per_epoch': seconds_per_epoch, 'num_named_neurons': num_named_neurons,
    'num_worms': num_worms, 'loss': loss_type, 'seq_len': seq_len, 'num_samples': num_samples, 'hidden_size': hidden_size,
    'worm_timesteps': worm_timesteps, 'optimizer': optimizer, 'learn_rate': lr})

    # Plot the seconds per epoch vs the number of worms
    fig, ax = plt.subplots(figsize=(10, 5))
    # Use whitegrid style
    sns.set_style('whitegrid')
    sns.set_palette('tab10')
    # Plot the data using sns bar plot
    sns.barplot(data=df, x=key, y='seconds_per_epoch', ax=ax)
    # Set the x axis and y axis labels according to the key
    plt.xlabel(key_name)
    plt.ylabel('Seconds per epoch')
    # Set title according to the key
    plt.title('Seconds per epoch vs {}'.format(key_name))
    plt.tight_layout()
    # plt.show()
    plt.close()
    # Save the plot
    fig.savefig(os.path.join(exp_log_dir, 'scaling_laws', 'seconds_per_epoch_bar'), dpi=300)

    if key == 'num_worms' or key == 'seq_len' or key == 'num_samples' or key == 'hidden_size' or key == 'worm_timesteps':
        # Plot the seconds per epoch vs the number of worms
        fig, ax = plt.subplots(figsize=(10, 5))
        # Use whitegrid style
        sns.set_style('whitegrid')
        sns.set_palette('tab10')
        # Plot the data using sns scatterplot
        sns.scatterplot(data=df, x=key, y='seconds_per_epoch', ax=ax)
        # Regression line
        if log_scale:
            # Use log scale for the x axis and y axis
            ax.set_xscale('log')
            ax.set_yscale('log')
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(df[key]), np.log(df['seconds_per_epoch']))
            # Add the regression line to the plot using sns lineplot
            x = np.linspace(df[key].min(), df[key].max(), 1000)
            y = np.exp(intercept) * x ** slope
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(df[key], df['seconds_per_epoch'])
            # Add the regression line to the plot using sns lineplot
            x = np.linspace(df[key].min(), df[key].max(), 1000)
            y = slope * x + intercept
        # Write the regression line equation as a label
        label = 'y = {}x + {}\n~ R^2: {}'.format(round(slope, 5), round(intercept, 5), round(r_value ** 2, 5))
        sns.lineplot(x=x, y=y, ax=ax, label=label, color='red')
        # Use dashed line style
        ax.lines[0].set_linestyle("--")
        # Reduce the opacity of the regression line
        ax.lines[0].set_alpha(0.5)
        # Reduce size of the regression line
        ax.lines[0].set_linewidth(1)

        # Set the x axis and y axis labels according to the key
        plt.xlabel(key_name)
        plt.ylabel('Seconds per epoch')
        # Set title according to the key
        plt.title('Seconds per epoch vs {}'.format(key_name))
        plt.tight_layout()
        # plt.show()
        plt.close()
        # Save the plot inside the scaling_laws folder
        fig.savefig(os.path.join(exp_log_dir, 'scaling_laws', 'seconds_per_epoch_scatter'), dpi=300)


    # Return df, fig and ax so we can customize the plot if we want
    return df, fig, ax
