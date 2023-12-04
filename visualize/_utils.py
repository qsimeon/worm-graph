from visualize._pkg import *

# Init logger
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)  # Suppress matplotlib logging


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


def plot_dataset_info(log_dir):
    # Map the 302 C. elgans neurons to their index
    neuron_idx_mapping = {neuron: idx for idx, neuron in enumerate(NEURONS_302)}

    # Train dataset
    df_train = pd.read_csv(
        os.path.join(log_dir, "dataset", "train_dataset_info.csv"),
        converters={"neurons": ast.literal_eval},
    )
    neurons_train = df_train["neurons"]
    # Flatten the list of lists into a single list of neurons
    flattened_neurons_train = [
        neuron for sublist in neurons_train for neuron in sublist
    ]
    # Now use np.unique on this flattened list
    unique_neurons_train, neuron_counts_train = np.unique(
        flattened_neurons_train, return_counts=True
    )
    # Standard sorting
    standard_counts_train = np.zeros(302, dtype=int)
    neuron_idx = [neuron_idx_mapping[neuron] for neuron in unique_neurons_train]
    standard_counts_train[neuron_idx] = neuron_counts_train
    # Get unique datasets
    train_exp_datasets = df_train["dataset"].unique().tolist()
    # Create DataFrame for train data
    train_data = {"Neuron": NEURONS_302, "Count": standard_counts_train}
    df_train_plot = pd.DataFrame(train_data)

    # Validation dataset
    df_val = pd.read_csv(
        os.path.join(log_dir, "dataset", "val_dataset_info.csv"),
        converters={"neurons": ast.literal_eval},
    )
    neurons_val = df_val["neurons"]
    # Flatten the list of lists into a single list of neurons
    flattened_neurons_val = [neuron for sublist in neurons_val for neuron in sublist]
    # Now use np.unique on this flattened list
    unique_neurons_val, neuron_counts_val = np.unique(
        flattened_neurons_val, return_counts=True
    )
    # Standard sorting
    standard_counts_val = np.zeros(302, dtype=int)
    neuron_idx_val = [neuron_idx_mapping[neuron] for neuron in unique_neurons_val]
    standard_counts_val[neuron_idx_val] = neuron_counts_val
    # Get unique datasets
    val_exp_datasets = df_val["dataset"].unique().tolist()
    # Create DataFrame for validation data
    val_data = {"Neuron": NEURONS_302, "Count": standard_counts_val}
    df_val_plot = pd.DataFrame(val_data)

    # Plot histogram using sns
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    sns.set_style("whitegrid")
    sns.set_palette("tab10")

    # Train dataset plot
    sns.barplot(x="Neuron", y="Count", data=df_train_plot, ax=ax[0], errorbar=None)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha="right")
    ax[0].set_ylabel("Count", fontsize=12)
    ax[0].set_xlabel("Neuron", fontsize=12)
    ax[0].set_title("Neuron count of Train Dataset", fontsize=14)
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    metadata_train_text = (
        "Experimental datasets used: {}\nTotal number of worms: {}".format(
            ", ".join(train_exp_datasets), len(df_train)
        )
    )
    ax[0].text(
        0.02,
        0.95,
        metadata_train_text,
        transform=ax[0].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round, pad=1", facecolor="white", edgecolor="black", alpha=0.5
        ),
    )

    # Validation dataset plot
    sns.barplot(x="Neuron", y="Count", data=df_val_plot, ax=ax[1], errorbar=None)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha="right")
    ax[1].set_ylabel("Count", fontsize=12)
    ax[1].set_xlabel("Neuron", fontsize=12)
    ax[1].set_title("Neuron count of Validation Dataset", fontsize=14)
    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    metadata_val_text = (
        "Experimental datasets used: {}\nTotal number of worms: {}".format(
            ", ".join(val_exp_datasets), len(df_val)
        )
    )
    ax[1].text(
        0.02,
        0.95,
        metadata_val_text,
        transform=ax[1].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round, pad=1", facecolor="white", edgecolor="black", alpha=0.5
        ),
    )

    # Set y-axes to only use integer values
    ax[0].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax[1].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()

    # Save figure
    fig.savefig(os.path.join(log_dir, "dataset", "dataset_info.png"), dpi=300)
    plt.close()


def plot_loss_curves(log_dir, info_to_display=None):
    """Plots the loss curves stored in a log directory.

    Args:
        log_dir (str): The path to the log directory.
    """

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.set(style="whitegrid")
    sns.set_palette("tab10")

    # Load loss curves
    loss_curves_csv = os.path.join(log_dir, "train", "train_metrics.csv")
    if not os.path.exists(loss_curves_csv):
        logger.error("No loss curves found in the log directory.")
        return None

    loss_df = pd.read_csv(loss_curves_csv, index_col=0)

    sns.lineplot(
        x="epoch",
        y="train_baseline",
        data=loss_df,
        ax=ax,
        label="Train baseline",
        color="c",
        alpha=0.8,
        errorbar=None,
        **dict(linestyle=":"),
    )

    sns.lineplot(
        x="epoch",
        y="val_baseline",
        data=loss_df,
        ax=ax,
        label="Validation baseline",
        color="r",
        alpha=0.8,
        errorbar=None,
        **dict(linestyle=":"),
    )
    sns.lineplot(
        x="epoch",
        y="train_loss",
        data=loss_df,
        ax=ax,
        label="Train",
        errorbar=None,
    )
    sns.lineplot(
        x="epoch",
        y="val_loss",
        data=loss_df,
        ax=ax,
        label="Validation",
        errorbar=None,
    )

    # Set x-axis to only use integer values
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Legend and title
    plt.legend(frameon=True, loc="upper right", fontsize=12)
    plt.title("Learning curves", fontsize=16)

    # Do some repositioning
    x_position_percent = 0.075  # Adjust this value to set the desired position
    x_position_box = (
        ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * x_position_percent
    )
    y_position_percent = 0.80  # Adjust this value to set the desired position
    y_position_box = (
        ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * y_position_percent
    )
    if info_to_display is not None:
        plt.text(
            x_position_box,
            y_position_box,
            info_to_display,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
            style="italic",
        )
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(log_dir, "train", "loss_curves.png"), dpi=300)
    plt.close()

    return None


def plot_predictions(log_dir, neurons_to_plot=None, worms_to_plot=None):
    for type_ds in os.listdir(os.path.join(log_dir, "prediction")):
        for ds_name in os.listdir(os.path.join(log_dir, "prediction", type_ds)):
            # Get the list of worms
            worm_list = os.listdir(
                os.path.join(log_dir, "prediction", type_ds, ds_name)
            )

            # Treat worms_to_plot
            if isinstance(worms_to_plot, int):
                # If worms_to_plot is an integer, randomly select that many worms
                worm_list = np.random.choice(
                    worm_list, size=min(worms_to_plot, len(worm_list)), replace=False
                ).tolist()
            elif isinstance(worms_to_plot, list):
                # If it is a list, filter out any worms not in worms_to_plot
                worm_list = [worm for worm in worm_list if worm in worms_to_plot]
            elif isinstance(worms_to_plot, str):
                # If worms_to_plot is a str, keep only the requested wormID
                worm_list = [worm for worm in worm_list if worm == worms_to_plot]

            # Iterate over the worms
            for wormID in worm_list:
                url = os.path.join(
                    log_dir, "prediction", type_ds, ds_name, wormID, "predictions.csv"
                )
                neurons_url = os.path.join(
                    log_dir, "prediction", type_ds, ds_name, wormID, "named_neurons.csv"
                )

                # Access the prediction directory
                df = pd.read_csv(url)
                df.set_index(["Type", "Unnamed: 1"], inplace=True)
                df.index.names = ["Type", ""]

                # Get the named neurons
                neurons_df = pd.read_csv(
                    neurons_url,
                )
                neurons = neurons_df["named_neurons"]

                # Treat neurons_to_plot
                if isinstance(neurons_to_plot, int):
                    # Randomly select number of neurons
                    neurons = np.random.choice(
                        neurons, size=min(neurons_to_plot, len(neurons)), replace=False
                    ).tolist()
                elif isinstance(neurons_to_plot, list):
                    # Filter out neurons not listed
                    neurons = [
                        neuron for neuron in neurons_to_plot if neuron in neurons
                    ]
                elif isinstance(neurons_to_plot, str):
                    # Get only the requested neuron
                    neurons = [
                        neuron for neuron in neurons if neuron == neurons_to_plot
                    ]

                # Get the time vectors
                data_window = len(
                    pd.concat([df.loc["Context"], df.loc["Ground Truth"]], axis=0)
                )
                generate_window = len(
                    pd.concat([df.loc["Context"], df.loc["AR Generation"]], axis=0)
                )
                time_vector = np.arange(generate_window)

                time_context = time_vector[: len(df.loc["Context"])]
                time_ground_truth = time_vector[
                    len(df.loc["Context"]) - 1 : data_window - 1
                ]
                # -1 for plot continuity
                time_gt_generated = time_vector[len(df.loc["Context"]) - 1 : -1]
                time_ar_generated = time_vector[len(df.loc["Context"]) - 1 : -1]

                sns.set_style("whitegrid")

                palette = sns.color_palette("tab10")
                gt_color = palette[0]  # Blue
                gt_generation_color = palette[
                    1
                ]  # orange (next time step prediction with gt)
                ar_generation_color = palette[
                    2
                ]  # green (autoregressive next time step prediction)

                # Metadata textbox
                metadata_text = "Dataset: {}\nWorm ID: {}".format(ds_name, wormID)

                for neuron in neurons:
                    fig, ax = plt.subplots(figsize=(10, 4))

                    ax.plot(
                        time_context,
                        df.loc["Context", neuron],
                        color=gt_color,
                        label="Ground truth",
                    )
                    ax.plot(
                        time_ground_truth,
                        df.loc["Ground Truth", neuron],
                        color=gt_color,
                        alpha=0.9,
                    )

                    ax.plot(
                        time_gt_generated,
                        df.loc["GT Generation", neuron],
                        color=gt_generation_color,
                        label="'Ground truth feeding'",
                    )
                    ax.plot(
                        time_ar_generated,
                        df.loc["AR Generation", neuron],
                        alpha=0.5,
                        color=ar_generation_color,
                        label="Autoregressive",
                    )

                    # Fill the context window
                    ax.axvspan(
                        time_context[0],
                        time_context[-1],
                        alpha=0.1,
                        color=gt_color,
                        label="Initial context window",
                    )

                    ax.set_title(f"Calcium Activity of Neuron {neuron}")
                    ax.set_xlabel("Time steps")
                    ax.set_ylabel("Activity ($\Delta F / F$)")
                    ax.legend(loc="upper right")

                    # Add metadata textbox in upper left corner
                    ax.text(
                        0.02,
                        0.95,
                        metadata_text,
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment="top",
                        bbox=dict(
                            boxstyle="round, pad=1",
                            facecolor="white",
                            edgecolor="black",
                            alpha=0.5,
                        ),
                    )

                    plt.tight_layout()

                    # Make figure directory
                    os.makedirs(
                        os.path.join(
                            log_dir, "prediction", type_ds, ds_name, wormID, "neurons"
                        ),
                        exist_ok=True,
                    )

                    # Save figure
                    plt.savefig(
                        os.path.join(
                            log_dir,
                            "prediction",
                            type_ds,
                            ds_name,
                            wormID,
                            "neurons",
                            f"{neuron}.png",
                        ),
                        dpi=300,
                    )
                    plt.close()


def plot_pca_trajectory(log_dir, worms_to_plot=None, plot_type="3D"):
    for type_ds in os.listdir(os.path.join(log_dir, "prediction")):
        for ds_name in os.listdir(os.path.join(log_dir, "prediction", type_ds)):
            # Get the list of worms
            worm_list = os.listdir(
                os.path.join(log_dir, "prediction", type_ds, ds_name)
            )

            # If worms_to_plot is an integer, randomly select that many worms
            if isinstance(worms_to_plot, int):
                worm_list = np.random.choice(
                    worm_list, size=min(worms_to_plot, len(worm_list)), replace=False
                ).tolist()
            elif isinstance(worms_to_plot, list):
                # Filter out the worms not in worms_to_plot
                worm_list = [worm for worm in worm_list if worm in worms_to_plot]
            elif isinstance(worms_to_plot, str):
                worm_list = [worm for worm in worm_list if worm == worms_to_plot]
            # If worms_to_plot is None, keep the entire worm_list

            for wormID in worm_list:
                url = os.path.join(
                    log_dir, "prediction", type_ds, ds_name, wormID, "predictions.csv"
                )
                neurons_url = os.path.join(
                    log_dir, "prediction", type_ds, ds_name, wormID, "named_neurons.csv"
                )

                df = pd.read_csv(url)

                # Get the named neurons
                neurons_df = pd.read_csv(
                    neurons_url,
                )
                neurons = neurons_df["named_neurons"]

                sns.set_style("whitegrid")
                palette = sns.color_palette("tab10")
                gt_color = palette[0]  # Blue
                gt_generation_color = palette[
                    1
                ]  # orange (next time step prediction with gt)
                ar_generation_color = palette[
                    2
                ]  # green (autoregressive next time step prediction)

                # Split data by Type
                ar_gen_data = df[df["Type"] == "AR Generation"].drop(
                    columns=["Type", "Unnamed: 1"]
                )
                ar_gen_data = ar_gen_data[neurons]  # Filter only named neurons

                ground_truth_data = df[df["Type"] == "Ground Truth"].drop(
                    columns=["Type", "Unnamed: 1"]
                )
                ground_truth_data = ground_truth_data[
                    neurons
                ]  # Filter only named neurons

                # Extract GT Generation data
                gt_gen_data = df[df["Type"] == "GT Generation"].drop(
                    columns=["Type", "Unnamed: 1"]
                )
                gt_gen_data = gt_gen_data[neurons]  # Filter only named neurons

                # Combine and Standardize the data
                all_data = pd.concat([ar_gen_data, ground_truth_data, gt_gen_data])
                scaler = StandardScaler()
                standardized_data = scaler.fit_transform(all_data)

                try:
                    # Apply PCA
                    if plot_type == "2D":
                        pca = PCA(n_components=2)
                    else:
                        pca = PCA(n_components=3)
                    reduced_data = pca.fit_transform(standardized_data)

                    # Plot
                    if plot_type == "2D":
                        plt.figure(figsize=(8, 7))

                        plt.plot(
                            reduced_data[: len(ar_gen_data), 0],
                            reduced_data[: len(ar_gen_data), 1],
                            alpha=0.5,
                            color=ar_generation_color,
                            label="Autoregressive",
                            linestyle="-",
                            marker="o",
                        )
                        plt.plot(
                            reduced_data[
                                len(ar_gen_data) : len(ar_gen_data)
                                + len(ground_truth_data),
                                0,
                            ],
                            reduced_data[
                                len(ar_gen_data) : len(ar_gen_data)
                                + len(ground_truth_data),
                                1,
                            ],
                            color=gt_color,
                            label="Ground Truth",
                            linestyle="-",
                            marker="o",
                        )
                        plt.plot(
                            reduced_data[
                                len(ar_gen_data) + len(ground_truth_data) :, 0
                            ],
                            reduced_data[
                                len(ar_gen_data) + len(ground_truth_data) :, 1
                            ],
                            color=gt_generation_color,
                            label="'Ground truth feeding'",
                            linestyle="-",
                            marker="o",
                        )

                        # Mark starting points with black stars
                        plt.scatter(
                            reduced_data[0, 0],
                            reduced_data[0, 1],
                            color="black",
                            marker="*",
                            s=50,
                        )
                        plt.scatter(
                            reduced_data[len(ar_gen_data), 0],
                            reduced_data[len(ar_gen_data), 1],
                            color="black",
                            marker="*",
                            s=50,
                        )
                        plt.scatter(
                            reduced_data[len(ar_gen_data) + len(ground_truth_data), 0],
                            reduced_data[len(ar_gen_data) + len(ground_truth_data), 1],
                            color="black",
                            marker="*",
                            s=50,
                        )

                        plt.xlabel("Principal Component 1")
                        plt.ylabel("Principal Component 2")

                        # Text box with PCA explained variance
                        textstr = "\n".join(
                            (
                                r"$PC_1=%.2f$" % (pca.explained_variance_ratio_[0],),
                                r"$PC_2=%.2f$" % (pca.explained_variance_ratio_[1],),
                            )
                        )
                        props = dict(boxstyle="round", facecolor="white", alpha=0.5)
                        plt.text(
                            0.05,
                            0.95,
                            textstr,
                            transform=plt.gca().transAxes,
                            fontsize=14,
                            verticalalignment="top",
                            bbox=props,
                        )

                    else:
                        fig = plt.figure(figsize=(8, 7))
                        ax = fig.add_subplot(111, projection="3d")

                        ax.plot(
                            reduced_data[: len(ar_gen_data), 0],
                            reduced_data[: len(ar_gen_data), 1],
                            reduced_data[: len(ar_gen_data), 2],
                            alpha=0.5,
                            color=ar_generation_color,
                            label="Autoregressive",
                            linestyle="-",
                            marker="o",
                        )
                        ax.plot(
                            reduced_data[
                                len(ar_gen_data) : len(ar_gen_data)
                                + len(ground_truth_data),
                                0,
                            ],
                            reduced_data[
                                len(ar_gen_data) : len(ar_gen_data)
                                + len(ground_truth_data),
                                1,
                            ],
                            reduced_data[
                                len(ar_gen_data) : len(ar_gen_data)
                                + len(ground_truth_data),
                                2,
                            ],
                            color=gt_color,
                            label="Ground Truth",
                            linestyle="-",
                            marker="o",
                        )
                        ax.plot(
                            reduced_data[
                                len(ar_gen_data) + len(ground_truth_data) :, 0
                            ],
                            reduced_data[
                                len(ar_gen_data) + len(ground_truth_data) :, 1
                            ],
                            reduced_data[
                                len(ar_gen_data) + len(ground_truth_data) :, 2
                            ],
                            color=gt_generation_color,
                            label="'Ground truth feeding'",
                            linestyle="-",
                            marker="o",
                        )

                        # Mark starting points with black stars
                        ax.scatter(
                            reduced_data[0, 0],
                            reduced_data[0, 1],
                            reduced_data[0, 2],
                            color="black",
                            marker="*",
                            s=50,
                        )
                        ax.scatter(
                            reduced_data[len(ar_gen_data), 0],
                            reduced_data[len(ar_gen_data), 1],
                            reduced_data[len(ar_gen_data), 2],
                            color="black",
                            marker="*",
                            s=50,
                        )
                        ax.scatter(
                            reduced_data[len(ar_gen_data) + len(ground_truth_data), 0],
                            reduced_data[len(ar_gen_data) + len(ground_truth_data), 1],
                            reduced_data[len(ar_gen_data) + len(ground_truth_data), 2],
                            color="black",
                            marker="*",
                            s=50,
                        )

                        ax.set_xlabel("Principal Component 1")
                        ax.set_ylabel("Principal Component 2")
                        ax.set_zlabel("Principal Component 3")

                        # Text box with PCA explained variance
                        textstr = "\n".join(
                            (
                                r"$PC_1=%.2f$" % (pca.explained_variance_ratio_[0],),
                                r"$PC_2=%.2f$" % (pca.explained_variance_ratio_[1],),
                                r"$PC_3=%.2f$" % (pca.explained_variance_ratio_[2],),
                            )
                        )
                        props = dict(boxstyle="round", facecolor="white", alpha=0.5)
                        ax.text(
                            0.0,
                            0.0,
                            0.0,
                            textstr,
                            transform=ax.transAxes,
                            fontsize=14,
                            verticalalignment="bottom",
                            bbox=props,
                        )

                    plt.legend()
                    plt.title(f"PCA Trajectories of Predictions in {plot_type}")
                    plt.tight_layout()

                    # Make figure directory
                    os.makedirs(
                        os.path.join(
                            log_dir, "prediction", type_ds, ds_name, wormID, "pca"
                        ),
                        exist_ok=True,
                    )

                    # Save figure
                    plt.savefig(
                        os.path.join(
                            log_dir,
                            "prediction",
                            type_ds,
                            ds_name,
                            wormID,
                            "pca",
                            f"pca_{plot_type}.png",
                        ),
                        dpi=300,
                    )
                    plt.close()

                except Exception as e:
                    logger.info(
                        f"PCA plot failed for {plot_type} in {type_ds} dataset (check if num_named_neurons >= 3)"
                    )
                    logger.error(f"The error that occurred: {e}")
                    logger.error(
                        traceback.format_exc()
                    )  # This will print the full traceback
                    pass


def plot_worm_data(worm_data, num_neurons=5, smooth=False):
    """
    Plot a few calcium traces from a given worm's data.

    :param worm_data: The data for a single worm.
    :param num_neurons: The number of neurons to plot.
    """

    np.random.seed(42)  # random seed for reproducibility

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
    plt.ylabel("Calcium Activity")
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


def experiment_parameter(exp_dir, key):
    """
    Returns a tuple containing the value, title, and x-axis label for a given experiment parameter.

    Parameters:
    exp_dir (str): The path to the experiment directory.
    key (str): The name of the experiment parameter to retrieve.
        Options for the different experiment parameters (key) are:
        - experiment_seed: The random seed used for the experiment
        - num_worms: The number of worms in the trainining set
        - num_time_steps: The total number of train time steps
        - num_named_neurons: The number of distinct named neurons recorded across all worms
        - time_steps_per_neuron: The average number of train time steps per neuron
        - num_samples: The number of training sequences sampled per worm
        - hidden_size: The hidden size of the model
        - batch_size: The batch size used for training
        - seq_len: The sequence length used for training
        - learn_rate: The learning rate used for training
        - dataset: The name of the dataset used for training (TODO)
        - model: The type of neural net model used for training
        - optimizer: The type of optimizer used for training
        - time_last_epoch: The computation time in seconds for the last epoch
        - computation_flops: The number of floating point operations (FLOPs)
        - num_parameters: The total number of trainable parameters in the model

    Returns:
    tuple: A tuple containing the value, title, and x-axis label for the experiment parameter.
    """
    # Set some default values
    value = exp_dir.split("/")[-1]  # exp<int> (default)
    title = "Experiment"
    xaxis = "Experiment run"

    if key == "experiment_seed" or key == "exp_seed" or key == "seed":
        # The random seed used for the experiment
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.experiment.seed
        title = "Experiment random seed"
        xaxis = "Seed"

    if key == "num_worms" or key == "num_train_worms":
        # The number of worms in the training set
        df = pd.read_csv(
            os.path.join(exp_dir, "dataset", "train_dataset_info.csv"),
            converters={"neurons": ast.literal_eval},
        )
        value = df["combined_dataset_index"].nunique()
        title = "Number of worms in training set"
        xaxis = "Num. worms"

    if key == "num_time_steps":
        # Total number of train time steps in the dataset
        df = pd.read_csv(
            os.path.join(exp_dir, "dataset", "train_dataset_info.csv"),
            converters={"neurons": ast.literal_eval},
        )
        value = df["train_time_steps"].sum()
        title = "Total amount of training data"
        xaxis = "Num. time steps"

    if key == "num_neurons" or key == "num_named_neurons":
        # Number of named neurons used for training
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        df = pd.read_csv(
            os.path.join(exp_dir, "dataset", "train_dataset_info.csv"),
            converters={"neurons": ast.literal_eval},
        )
        value = pipeline_info.submodule.dataset.num_named_neurons
        title = "Number of unique labelled neurons"
        xaxis = "Num. neurons"
        if value is None:
            # Create a set of all unique neurons
            unique_neurons = set()
            for neuron_list in df["neurons"]:
                unique_neurons.update(neuron_list)
                value = len(unique_neurons)
        title = "Number of unique labelled neurons"
        xaxis = "Num. neurons"

    if key == "time_steps_per_neuron":
        # Average number of train time steps per neurons
        df = pd.read_csv(
            os.path.join(exp_dir, "dataset", "train_dataset_info.csv"),
            converters={"neurons": ast.literal_eval},
        )
        value = (df["train_time_steps"] / df["num_neurons"]).median()
        title = "Average amount of training data per neuron"
        xaxis = "Num. time steps per neuron"

    if key == "num_train_samples" or key == "num_samples":
        # The number of training sequences sampled per worm
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.submodule.dataset.num_train_samples
        title = "Number of training samples"
        xaxis = "Num. training samples"

    if key == "hidden_size":
        # The hidden size of the neural network model
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.submodule.model.hidden_size
        title = "Hidden size"
        xaxis = "Hidden size"

    if key == "batch_size":
        # The batch size used for training the model
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.submodule.train.batch_size
        title = "Batch size"
        xaxis = "Batch size"

    if key == "seq_len":
        # Sequence length used for training the models
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.submodule.dataset.seq_len
        title = "Sequence length"
        xaxis = "Sequence length"

    if key == "lr" or key == "learn_rate" or key == "learning_rate":
        # The learning rate used for training the model
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.submodule.train.lr  # Learning rate used for training
        title = "Learning rate"
        xaxis = "Learning rate"

    if key == "dataset" or key == "dataset_name" or key == "worm_dataset":
        # The name of the single dataset used for training
        pass

    if key == "model" or key == "model_type":
        # The type of neural network model used
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.submodule.model.type
        title = "Model"
        xaxis = "Model type"

    if key == "optimizer" or key == "optimizer_type":
        # The type of optimizer used for training the model
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.submodule.train.optimizer
        title = "Optimizer"
        xaxis = "Optimizer type"

    if key == "loss" or key == "loss_type":
        # The type of loss function used for training
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.submodule.model.loss
        title = "Loss function"
        xaxis = "Loss function type"

    if key == "time_last_epoch" or key == "computation_time":
        # The computation time in seconds for the last epoch
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        chkpt_path = os.path.join(exp_dir, "train", "checkpoints", f"model_best.pt")
        model_chkpt = torch.load(chkpt_path, map_location=torch.device(DEVICE))
        value = model_chkpt["time_last_epoch"]  # TODO: Fix! should be `time_last_epoch`
        title = "Computation time of last epoch"
        xaxis = "Time (s)"

    if key == "computation_flops" or key == "flops":
        # The number of floating point operations (FLOPs) for the model
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        chkpt_path = os.path.join(exp_dir, "train", "checkpoints", f"model_best.pt")
        model_chkpt = torch.load(chkpt_path, map_location=torch.device(DEVICE))
        value = model_chkpt["computation_flops"]
        title = "Computation floating point operations"
        xaxis = "FLOPs"

    if key == "num_parameters" or key == "num_params" or key == "num_trainable_params":
        # The total number of trainable parameters in the model
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        chkpt_path = os.path.join(exp_dir, "train", "checkpoints", f"model_best.pt")
        model_chkpt = torch.load(chkpt_path, map_location=torch.device(DEVICE))
        value = model_chkpt["num_trainable_params"]
        title = "Number of trainable parameters"
        xaxis = "Num. trainable parameters"

    return value, title, xaxis


def plot_experiment_losses(exp_log_dir, exp_key, exp_plot_dir=None):
    """
    Plot train and validation loss curves and baseline losses for all experiments.

    Args:
    - exp_log_dir (str): path to directory containing experiment logs
    - exp_key (str): key to identify the experiment
    - exp_plot_dir (str or None): path to directory to save the plot.

    Returns:
    - None
    """

    # Store parameters, epochs and losses for plotting
    parameters = []
    epochs = []
    train_losses = []
    train_baselines = []
    val_losses = []
    val_baselines = []

    # Loop over trials/repetitions of the experiment
    for file in sorted(os.listdir(exp_log_dir), key=lambda x: x.strip("exp_")):
        # Skip if not in a trial/repetition directory
        if not file.startswith("exp") or file.startswith("exp_"):
            continue

        # Get experiment directory
        exp_dir = os.path.join(exp_log_dir, file)

        # Load train metrics if avalibale
        metrics_csv = os.path.join(exp_dir, "train", "train_metrics.csv")
        if os.path.exists(metrics_csv):
            # Get epochs and loss values
            df = pd.read_csv(metrics_csv)

            # Store all loss values to be plotted
            epochs.append(df["epoch"])
            train_losses.append(df["train_loss"])
            train_baselines.append(df["train_baseline"])
            val_losses.append(df["val_loss"])
            val_baselines.append(df["val_baseline"])

            # Get parameter values
            exp_param, exp_title, exp_xaxis = experiment_parameter(exp_dir, key=exp_key)
            parameters.append(exp_param)

        # Skip otherwise
        else:
            continue

    # Get indices for sorting parameters and sort all the lists
    sorted_indices = np.argsort(parameters)
    parameters = [parameters[i] for i in sorted_indices]
    epochs = [epochs[i] for i in sorted_indices]
    train_losses = [train_losses[i] for i in sorted_indices]
    train_baselines = [train_baselines[i] for i in sorted_indices]
    val_losses = [val_losses[i] for i in sorted_indices]
    val_baselines = [val_baselines[i] for i in sorted_indices]

    # Plot all trials/repetitions of the experiment
    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    sns.set_style("whitegrid")

    # Normalize the exp_param values for colormap
    norm = Normalize(vmin=min(parameters), vmax=max(parameters))
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cm.YlOrRd)

    # Loop over parameter values and plot losses
    for i, param_val in enumerate(parameters):
        # Get color for current parameter value
        color_val = scalar_map.to_rgba(param_val)

        # Plot loss and baseline
        ax[0].plot(
            epochs[i],
            train_losses[i],
            marker=".",
            label=param_val,
            color=color_val,
        )
        ax[0].plot(
            epochs[i],
            train_baselines[i],
            color="black",
            linestyle="--",
        )
        ax[1].plot(
            epochs[i],
            val_losses[i],
            marker=".",
            label=param_val,
            color=color_val,
        )
        ax[1].plot(
            epochs[i],
            val_baselines[i],
            color="black",
            linestyle="--",
        )

    # Set x-axes to only use integer values
    ax[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Set y-axes to use log-scale
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")

    # Set loss labels
    ax[0].set_xlabel("Epoch", fontsize=12)
    ax[0].set_ylabel("Train loss", fontsize=12)
    ax[1].set_xlabel("Epoch", fontsize=12)
    ax[1].set_ylabel("Validation loss", fontsize=12)

    # Get handles and labels for the legend
    handles, labels = ax[0].get_legend_handles_labels()

    # Randomly sample a subset of handles and labels (up to 10)
    num_samples = min(10, len(handles))  # Ensure not to sample more than available
    sampled_indices = np.random.choice(len(handles), num_samples, replace=False)
    sampled_handles = [handles[i] for i in sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]

    # Sort the labels and then sort handles accordingly
    sampled_labels, sampled_handles = zip(
        *sorted(
            zip(sampled_labels, sampled_handles),
            key=lambda t: float(t[0])
            if (isinstance(t[0], str) and str(t[0]).isnumeric())
            else t[0],
        )
    )

    # Set the legend with the sampled subset
    legend = ax[0].legend(sampled_handles, sampled_labels, fontsize=10, loc="best")
    legend.set_title(exp_xaxis)

    # Set loss figure title
    fig.suptitle(f"{exp_title} experiment", fontsize=14)
    plt.tight_layout()

    # Save or display the plot
    if exp_plot_dir:
        fig.savefig(os.path.join(exp_plot_dir, "exp_loss_curves.png"), dpi=300)
        plt.close()
    else:  # exp_plot_dir is None
        plt.show()

    # Return the figure and axes
    return fig, ax


def plot_experiment_summaries(exp_log_dir, exp_key, exp_plot_dir=None):
    """
    Plots summaries of experiments given a directory of experiment logs.

    Args:
        exp_log_dir (str): Path to directory containing experiment logs.
        exp_key (str): Key to identify the experiment parameter to plot.
        exp_plot_dir (str, optional): Path to directory to save the plot. Defaults to None.

    Returns:
        tuple: A tuple containing the figure and axes objects of the plot.
    """

    # Store parameters and losses for plotting
    parameters = []
    train_losses = []
    train_baselines = []
    val_losses = []
    val_baselines = []
    computation_times = []
    computation_flops = []

    # Loop over trials/repetitions of the experiment
    for file in sorted(os.listdir(exp_log_dir), key=lambda x: x.strip("exp_")):
        # Skip if not in a trial/repetition directory
        if not file.startswith("exp") or file.startswith("exp_"):
            continue

        # Get experiment directory
        exp_dir = os.path.join(exp_log_dir, file)

        # Load train metrics and best/final model checkpoint if both avalibale
        metrics_csv = os.path.join(exp_dir, "train", "train_metrics.csv")
        best_model_ckpt = os.path.join(exp_dir, "train", "checkpoints", "model_best.pt")
        if os.path.exists(metrics_csv) and os.path.exists(best_model_ckpt):
            # Get loss values
            df = pd.read_csv(metrics_csv)

            # Model achieving minimum validation loss
            best_model = torch.load(best_model_ckpt, map_location=torch.device(DEVICE))

            # Store all summary statistics to be plotted
            train_losses.append(df["train_loss"].min())
            train_baselines.append(df["train_baseline"].median())
            val_losses.append(df["val_loss"].min())
            val_baselines.append(df["val_baseline"].median())
            computation_times.append(df["computation_time"].tolist())
            computation_flops.append(best_model["computation_flops"])

            # Get parameter values
            exp_param, exp_title, exp_xaxis = experiment_parameter(exp_dir, key=exp_key)
            parameters.append(exp_param)

        # Skip otherwise
        else:
            continue

    # Get indices for sorting parameters and sort all the lists
    sorted_indices = np.argsort(parameters)
    parameters = [parameters[i] for i in sorted_indices]
    train_losses = [train_losses[i] for i in sorted_indices]
    train_baselines = [train_baselines[i] for i in sorted_indices]
    val_losses = [val_losses[i] for i in sorted_indices]
    val_baselines = [val_baselines[i] for i in sorted_indices]
    computation_times = [computation_times[i] for i in sorted_indices]
    computation_flops = [computation_flops[i] for i in sorted_indices]

    # Plot figure summarizing various experiment metrics
    # Create figure
    fig, axes = plt.subplots(
        1, 4, figsize=(20, 5)
    )  # Adjust the figsize to fit all subplots
    param_range = range(len(parameters))  # numeric range for plots

    # Validation loss bar plot
    axes[0].bar(param_range, val_losses, color="blue", label="Min Validation Loss")
    for baseline in val_baselines:
        axes[0].axhline(
            y=baseline, color="black", linestyle="--", label="Validation Baseline"
        )
    axes[0].set_xticks(param_range)
    axes[0].set_xticklabels(parameters, rotation=90, ha="right", fontsize=6)
    axes[0].set_title("Min Validation Loss")
    axes[0].set_xlabel(exp_xaxis)
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")

    # Training loss bar plot
    axes[1].bar(param_range, train_losses, color="orange", label="Min Training Loss")
    for baseline in train_baselines:
        axes[1].axhline(
            y=baseline, color="black", linestyle="--", label="Training Baseline"
        )
    axes[1].set_xticks(param_range)
    axes[1].set_xticklabels(parameters, rotation=90, ha="right", fontsize=6)
    axes[1].set_title("Min Training Loss")
    axes[1].set_xlabel(exp_xaxis)
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")

    # Epoch computation times boxplot
    axes[2].boxplot(computation_times, notch=True, sym="")  # , bootstrap=1000)
    axes[2].set_xticklabels(parameters, rotation=90, ha="right", fontsize=6)
    axes[2].set_title("Computation Time per Epoch")
    axes[2].set_xlabel(exp_xaxis)
    axes[2].set_ylabel("Time (seconds)")

    # FLOPs bar plot
    axes[3].bar(param_range, computation_flops, color="green", label="FLOPs")
    axes[3].set_xticks(param_range)
    axes[3].set_xticklabels(parameters, rotation=90, ha="right", fontsize=6)
    axes[3].set_title("Model FLOPs")
    axes[3].set_xlabel(exp_xaxis)
    axes[3].set_ylabel("FLOPs")
    axes[3].set_yscale("log")

    # Set summary figure title
    fig.suptitle(f"{exp_title} experiment", fontsize=14)
    plt.tight_layout()

    # Save or display the plot
    if exp_plot_dir:
        fig.savefig(os.path.join(exp_plot_dir, "exp_summary.png"), dpi=300)
        plt.close()
    else:  # exp_plot_dir is None
        plt.show()

    # Return the figure and axes
    return fig, axes


def plot_loss_per_dataset(log_dir, mode="validation"):
    # Load train/validation losses
    losses = pd.read_csv(
        os.path.join(log_dir, "analysis", f"{mode}_loss_per_dataset.csv")
    )
    losses = losses.dropna().reset_index(drop=True)

    # Train dataset names
    train_info = pd.read_csv(
        os.path.join(log_dir, "dataset", "train_dataset_info.csv"),
        converters={"neurons": ast.literal_eval},
    )
    train_dataset_names = train_info["dataset"].unique()

    sns.set_theme(style="whitegrid")
    sns.set_palette("tab10")
    palette = sns.color_palette()

    fig, ax = plt.subplots(figsize=(10, 4))

    bar_width = 0.35  # You can adjust this value for desired distance between bars

    # Bar positions for two sets of data
    index = np.arange(len(losses))
    bar1_positions = index
    bar2_positions = index + bar_width

    # First plot both model and baseline losses
    ax.bar(
        bar1_positions,
        losses[f"{mode}_loss"],
        bar_width,
        label="Model",
        color=palette[0],
    )
    ax.bar(
        bar2_positions,
        losses[f"{mode}_baseline"],
        bar_width,
        label="Baseline",
        color=palette[1],
        alpha=0.4,
    )

    ax.set_xticks(
        index + bar_width / 2
    )  # Set x-ticks to be in the middle of the grouped bars
    ax.set_xticklabels(losses["dataset"].values, rotation=0, ha="center")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")  # Set y-axis to log-scale
    ax.set_title(f"{mode.capitalize()} set loss across datasets")
    ax.legend(loc="upper right")

    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    textstr = "Datasets used for training: \n{}".format(", ".join(train_dataset_names))
    ax.text(
        0.02,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    for i, v in enumerate(losses["num_worms"]):
        ax.text(
            i + bar_width / 2,  # Adjusted x-position to align the text
            max(losses.loc[i, [f"{mode}_loss", f"{mode}_baseline"]]),
            r"$n_{val} = $" + str(int(v)),
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()

    # Save figure
    plt.savefig(
        os.path.join(log_dir, "analysis", f"{mode}_loss_per_dataset.png"), dpi=300
    )
    plt.close()


def plot_experiment_loss_per_dataset(
    exp_log_dir, exp_key, exp_plot_dir=None, mode="validation"
):
    """
    This function plots the experiment loss per dataset.
    It collects information from experiment log directories and loads the losses per dataset.
    It then plots the loss vs. experiment parameter for each dataset.
    """
    # =============== Collect information ===============
    # Create an empty DataFrame to store the losses
    losses = pd.DataFrame(
        columns=["dataset", f"{mode}_loss", f"{mode}_baseline", "exp_param"]
    )

    # Loop through all experiments
    for file in sorted(os.listdir(exp_log_dir), key=lambda x: x.strip("exp_")):
        # Skip if not starts with exp
        if not file.startswith("exp") or file.startswith("exp_"):
            continue

        # Get experiment directory
        exp_dir = os.path.join(exp_log_dir, file)

        # Experiment parameters
        exp_param, exp_title, exp_xaxis = experiment_parameter(exp_dir, key=exp_key)

        # Load losses per dataset
        tmp_df = pd.read_csv(
            os.path.join(exp_dir, "analysis", f"{mode}_loss_per_dataset.csv")
        )

        # Add experiment parameter to dataframe
        tmp_df["exp_param"] = exp_param

        # Load train information
        train_info = pd.read_csv(
            os.path.join(exp_dir, "dataset", "train_dataset_info.csv"),
            converters={"neurons": ast.literal_eval},
        )

        # Dataset names used for training
        train_dataset_names = train_info["dataset"].unique()
        tmp_df["train_dataset_names"] = ", ".join(train_dataset_names)

        # Name of the model
        model_name = torch.load(
            os.path.join(exp_dir, "train", "checkpoints", "model_best.pt")
        )["model_name"]
        tmp_df["model_name"] = model_name

        # Append to dataframe
        losses = pd.concat([losses, tmp_df], axis=0)

    # Make sure all NaNs are dropped before setting the index
    losses = losses.dropna()  # Do not use reset_index here

    # Now set the multi-index with 'exp_param' and 'dataset'
    losses.set_index(["exp_param", "dataset"], inplace=True)

    # After this, your unique call should work correctly
    # Create one subplot per dataset, arranged in two columns
    num_datasets = len(losses.index.unique(level="dataset"))
    num_rows = int(np.ceil(num_datasets / 2))

    # =============== Start plotting ===============

    # Plot loss vs. exp_param (comparison)
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.set_style("whitegrid")
    sns.set_palette("tab10")

    # Plot loss vs. exp_param for all datasets
    palette = sns.color_palette("tab10", len(losses.index.unique(level="dataset")))
    for color_idx, dataset in enumerate(losses.index.unique(level="dataset")):
        # Get the subset of losses and baselines for the current dataset
        df_subset_model = losses.loc[
            losses.index.get_level_values("dataset") == dataset, f"{mode}_loss"
        ].reset_index()
        df_subset_baseline = losses.loc[
            losses.index.get_level_values("dataset") == dataset, f"{mode}_baseline"
        ].reset_index()

        # Get the model name for the current dataset
        model_name = losses.loc[
            losses.index.get_level_values("dataset") == dataset, "model_name"
        ].values[0]

        # Get the color for the current dataset
        color = palette[color_idx]

        # Scatter plot of model losses vs. experiment parameter
        sns.scatterplot(
            data=df_subset_model,
            x="exp_param",
            y=f"{mode}_loss",
            ax=ax,
            color=color,
            alpha=0.5,
            label=f"{model_name} ({dataset})",
        )

        # Line plot of baseline losses vs. experiment parameter
        sns.lineplot(
            data=df_subset_baseline,
            x="exp_param",
            y=f"{mode}_baseline",
            ax=ax,
            linestyle="--",
            color=color,
            errorbar=None,
        )

        # Annotate number of val. worms
        num_worms = losses.loc[
            losses.index.get_level_values("dataset") == dataset, "num_worms"
        ].values[0]
        min_exp_param = df_subset_baseline["exp_param"].min()
        max_baseline = df_subset_baseline[f"{mode}_baseline"].max()
        ax.annotate(
            r"$n_{val}=$" + f"{int(num_worms)}",
            (min_exp_param, max_baseline),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            fontsize=8,
            color=color,
        )

        # Log-log scale
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Try to fit linear regression (log-log)
        try:
            x = np.log(df_subset_model["exp_param"].values)
            y = np.log(df_subset_model[f"{mode}_loss"].values)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            fit_label = (
                "y = {:.2f}x + {:.1f}\n".format(slope, intercept)
                + r"$R^2=$"
                + "{:.2f}".format(r_value**2)
            )
            ax.plot(
                df_subset_model["exp_param"].values,
                np.exp(intercept + slope * x),
                linestyle="-",
                color=color,
                label=fit_label,
            )
        except Exception as e:
            logger.info(
                "Failed to fit linear regression (log-log scale) for dataset {}".format(
                    dataset
                )
            )
            logger.error(f"The error that occurred: {e}")
            logger.error(traceback.format_exc())  # This will print the full traceback
            pass

    # Set axis labels and title
    ax.set_xlabel(exp_xaxis)
    ax.set_ylabel("Loss")
    ax.set_yscale("log")  # Set y-axis to log-scale
    ax.set_title(f"{mode.capitalize()} set loss across datasets")
    ax.legend(loc="upper right", fontsize="x-small")

    plt.tight_layout()

    if exp_plot_dir is not None:
        plt.savefig(
            os.path.join(exp_plot_dir, f"{mode}_loss_per_dataset.png"),
            dpi=300,
        )
        plt.close()
    else:
        return fig, ax
