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
    neuron_idx_mapping = {neuron: idx for idx, neuron in enumerate(NEURONS_302)}

    # Train dataset
    df_train = pd.read_csv(os.path.join(log_dir, "dataset", "train_dataset_info.csv"))
    # Convert 'neurons' column to list
    df_train["neurons"] = df_train["neurons"].apply(lambda x: ast.literal_eval(x))
    # Get all neurons
    neurons_train, neuron_counts_train = np.unique(
        np.concatenate(df_train["neurons"].values), return_counts=True
    )
    # Standard sorting
    std_counts_train = np.zeros(302)
    neuron_idx = [neuron_idx_mapping[neuron] for neuron in neurons_train]
    std_counts_train[neuron_idx] = neuron_counts_train
    # Get unique datasets
    train_exp_datasets = df_train["dataset"].unique().tolist()

    # Validation dataset
    df_val = pd.read_csv(os.path.join(log_dir, "dataset", "val_dataset_info.csv"))
    # Convert 'neurons' column to list
    df_val["neurons"] = df_val["neurons"].apply(lambda x: ast.literal_eval(x))
    # Get all neurons
    neurons_val, neuron_counts_val = np.unique(
        np.concatenate(df_val["neurons"].values), return_counts=True
    )
    # Standard sorting
    std_counts_val = np.zeros(302)
    neuron_idx_val = [neuron_idx_mapping[neuron] for neuron in neurons_val]
    std_counts_val[neuron_idx_val] = neuron_counts_val
    # Get unique datasets
    val_exp_datasets = df_val["dataset"].unique().tolist()

    # Plot histogram using sns
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    sns.set_style("whitegrid")
    sns.set_palette("tab10")

    # Train dataset
    sns.barplot(x=NEURONS_302, y=std_counts_train, ax=ax[0])
    ax[0].set_xticklabels(NEURONS_302, rotation=45)
    ax[0].set_ylabel("Count", fontsize=12)
    ax[0].set_xlabel("Neuron", fontsize=12)
    ax[0].set_title("Neuron count of Train Dataset", fontsize=14)
    # Reduce the number of xticks for readability
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

    # Validation dataset
    sns.barplot(x=NEURONS_302, y=std_counts_val, ax=ax[1])
    ax[1].set_xticklabels(NEURONS_302, rotation=45)
    ax[1].set_ylabel("Count", fontsize=12)
    ax[1].set_xlabel("Neuron", fontsize=12)
    ax[1].set_title("Neuron count of Validation Dataset", fontsize=14)
    # Reduce the number of xticks for readability
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
        **dict(linestyle=":"),
    )
    sns.lineplot(x="epoch", y="train_loss", data=loss_df, ax=ax, label="Train")
    sns.lineplot(x="epoch", y="val_loss", data=loss_df, ax=ax, label="Validation")

    ax.xaxis.set_major_locator(
        ticker.MaxNLocator(integer=True)
    )  # Set x-axis to only use integer values

    plt.legend(frameon=True, loc="upper right", fontsize=12)
    plt.title("Learning curves", fontsize=16)

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

            logger.info(f"worms_to_plot:  {worms_to_plot}")

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

            logger.info(f"worm_list: {worm_list}")

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
                neurons_df = pd.read_csv(neurons_url)
                neurons = neurons_df["named_neurons"]

                logger.info(f"neurons_to_plot:  {neurons_to_plot}")

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

                logger.info(f"neurons: {neurons}")

                seq_len = len(
                    pd.concat([df.loc["Context"], df.loc["Ground Truth"]], axis=0)
                )
                max_time_steps = len(
                    pd.concat([df.loc["Context"], df.loc["AR Generation"]], axis=0)
                )
                time_vector = np.arange(max_time_steps)

                time_context = time_vector[: len(df.loc["Context"])]
                time_ground_truth = time_vector[
                    len(df.loc["Context"]) - 1 : seq_len - 1
                ]
                time_gt_generated = time_vector[
                    len(df.loc["Context"]) - 1 : seq_len - 1
                ]
                time_ar_generated = time_vector[
                    len(df.loc["Context"]) - 1 : max_time_steps - 1
                ]  # -1 for plot continuity

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
                        label="Ground truth activity",
                    )
                    ax.plot(
                        time_ground_truth,
                        df.loc["Ground Truth", neuron],
                        color=gt_color,
                        alpha=0.5,
                    )

                    ax.plot(
                        time_gt_generated,
                        df.loc["GT Generation", neuron],
                        color=gt_generation_color,
                        label="'Teacher forcing'",
                    )
                    ax.plot(
                        time_ar_generated,
                        df.loc["AR Generation", neuron],
                        color=ar_generation_color,
                        label="Autoregressive",
                    )

                    # Fill the context window
                    ax.axvspan(
                        time_context[0],
                        time_context[-1],
                        alpha=0.1,
                        color=gt_color,
                        label="Context window",
                    )

                    ax.set_title(f"Neuronal Activity of {neuron}")
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
                neurons_df = pd.read_csv(neurons_url)
                neurons = neurons_df["named_neurons"]

                sns.set_style("whitegrid")
                palette = sns.color_palette("tab10")
                gt_color = palette[0]  # Blue
                gt_generation_color = palette[
                    1
                ]  # orange (next time step prediction with gt)
                ar_generation_color = palette[
                    2
                ]  # gree (autoregressive next time step prediction)

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
                            label="'Teacher forcing'",
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
                            label="'Teacher forcing'",
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

                except:
                    logger.info(
                        f"PCA plot failed for {plot_type} in {type_ds} dataset (check if num_named_neurons >= 3)"
                    )
                    pass


def plot_worm_data(worm_data, num_neurons=5, smooth=False):
    """
    Plot a few calcium traces from a given worm's data.

    :param worm_data: The data for a single worm.
    :param num_neurons: The number of neurons to plot.
    """

    np.random.seed(42)  # set random seed for reproducibility

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


def experiment_parameter(exp_dir, key):
    """
    Returns a tuple containing the value, title, and x-axis label for a given experiment parameter.

    Parameters:
    exp_dir (str): The path to the experiment directory.
    key (str): The name of the experiment parameter to retrieve.

    Returns:
    tuple: A tuple containing the value, title, and x-axis label for the experiment parameter.
    """
    # Set some default values
    value = exp_dir.split("/")[-1]  # exp<int> (default)
    title = "Experiment"
    xaxis = "Experiment run"

    if key == "time_steps_volume":
        df = pd.read_csv(os.path.join(exp_dir, "dataset", "train_dataset_info.csv"))
        df["tsv"] = df["train_time_steps"] / df["num_neurons"]
        value = df["tsv"].sum()  # Total volume of train time steps
        title = "Volume of training data"
        xaxis = "Number of time steps / Number of Named Neurons"

    if key == "num_time_steps":
        df = pd.read_csv(os.path.join(exp_dir, "dataset", "train_dataset_info.csv"))
        value = df["train_time_steps"].sum()  # Total number of train time steps
        title = "Amount of training data"
        xaxis = "Number of time steps"

    if key == "num_parameters":
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        model = get_model(pipeline_info.submodule.model)
        total_params, total_trainable = print_parameters(model, verbose=False)
        value = total_trainable  # Total number of parameters
        title = "Number of trainable parameters"
        xaxis = "Number of trainable parameters"

    if key == "hidden_volume":
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        h_size = pipeline_info.submodule.model.hidden_size  # Model hidden dimension
        model = get_model(pipeline_info.submodule.model)
        total_params, total_trainable = print_parameters(model, verbose=False)
        value = h_size / total_trainable  # Total volume of hidden states
        title = "Volume of hidden states"
        xaxis = "Hidden dimension / Number of trainable parameters"

    if key == "hidden_size":
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.submodule.model.hidden_size  # Model hidden dimension
        title = "Hidden dimension"
        xaxis = "Hidden dimension"

    if key == "optimizer":
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.submodule.train.optimizer
        title = "Optimizer"
        xaxis = "Optimizer type"

    if key == "batch_size":
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.submodule.train.batch_size  # Experiment batch size
        title = "Batch size"
        xaxis = "Batch size"

    if key == "lr":
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.submodule.train.lr  # Learning rate used for training
        title = "Learning rate"
        xaxis = "Learning rate"

    if key == "num_named_neurons":
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = (
            pipeline_info.submodule.dataset.num_named_neurons
        )  # Number of named neurons used for training
        title = "Neuron population"
        xaxis = "Number of neurons"

    if key == "seq_len":
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = (
            pipeline_info.submodule.dataset.seq_len
        )  # Sequence length used for training
        title = "Sequence length"
        xaxis = "Sequence length"

    if key == "loss":
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.submodule.model.loss  # Loss function used for training
        title = "Loss function"
        xaxis = "Loss function type"

    if key == "num_train_samples":
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.submodule.dataset.num_train_samples
        title = "Number of training samples"
        xaxis = "Number of training samples"

    if key == "model_type":
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = pipeline_info.submodule.model.type  # Model type used for training
        title = "Model"
        xaxis = "Model type"

    if key == "num_train_samples":
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, "pipeline_info.yaml"))
        value = (
            pipeline_info.submodule.dataset.num_train_samples
        )  # Number of training samples used for training
        title = "Number of training samples"
        xaxis = "Number of training samples"

    if key == "computation_time":
        df = pd.read_csv(os.path.join(exp_dir, "train", "train_metrics.csv"))
        value = (
            df["computation_time"].min(),
            df["computation_time"].mean(),
            df["computation_time"].max(),
        )  # Computation time
        title = "Computation time"
        xaxis = "Computation time (s)"

    if key == "computation_flops":
        df = pd.read_csv(os.path.join(exp_dir, "train", "train_metrics.csv"))
        value = (
            df["computation_flops"].min(),
            df["computation_flops"].mean(),
            df["computation_flops"].max(),
        )  # Computation time
        title = "Computation FLOPs"
        xaxis = "FLOPs"

    return value, title, xaxis


def plot_experiment_losses(exp_log_dir, exp_plot_dir, exp_key):
    """
    Plot train and validation loss curves and baseline losses for all experiments.

    Args:
    - exp_log_dir (str): path to directory containing experiment logs
    - exp_plot_dir (str): path to directory to save the plot
    - exp_key (str): key to identify the experiment

    Returns:
    - None
    """

    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    sns.set_style("whitegrid")

    # Keep track of parameters
    parameters = []

    # Loop over all the experiment files
    for file in np.sort(os.listdir(exp_log_dir)):
        # Skip if not starts with exp
        if not file.startswith("exp") or file.startswith("exp_"):
            continue

        # Get experiment directory
        exp_dir = os.path.join(exp_log_dir, file)

        # Get and store parameters
        exp_param, exp_title, exp_xaxis = experiment_parameter(exp_dir, key=exp_key)
        parameters.append(exp_param)

        # Load train metrics
        df = pd.read_csv(os.path.join(exp_dir, "train", "train_metrics.csv"))

        # Plot loss and baseline
        ax[0].plot(df["epoch"], df["train_loss"], label=exp_param)
        ax[0].plot(df["epoch"], df["train_baseline"], color="black", linestyle="--")
        ax[1].plot(df["epoch"], df["val_loss"], label=exp_param)
        ax[1].plot(df["epoch"], df["val_baseline"], color="black", linestyle="--")

    # Set loss labels
    ax[0].set_xlabel("Epoch", fontsize=12)
    ax[0].set_ylabel("Train loss", fontsize=12)
    ax[1].set_xlabel("Epoch", fontsize=12)
    ax[1].set_ylabel("Validation loss", fontsize=12)

    # Set loss legend
    legend = ax[0].legend(fontsize=10, loc="best")
    legend.set_title(exp_xaxis)

    # Set loss title
    fig.suptitle(f"{exp_title} experiment", fontsize=14)
    plt.tight_layout()

    # Save figure
    fig.savefig(os.path.join(exp_plot_dir, "all_losses.png"))
    plt.close()


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
    # Initialize figure with subplots
    fig, axes = plt.subplots(
        1, 4, figsize=(20, 5)
    )  # Adjust the figsize to fit all subplots

    # Lists to store data
    val_losses = []
    train_losses = []
    computation_times = []
    flops = []
    exp_parameters = []

    # Lists to store baselines
    val_baselines = []
    train_baselines = []

    # Iterate over experiment directories
    for file in np.sort(os.listdir(exp_log_dir)):
        if not file.startswith("exp") or file.startswith("exp_"):
            continue

        exp_dir = os.path.join(exp_log_dir, file)
        metrics_csv = os.path.join(exp_dir, "train", "train_metrics.csv")

        if os.path.exists(metrics_csv):
            df = pd.read_csv(metrics_csv)

            # Append minimum losses and mean computation times
            val_losses.append(df["val_loss"].min())
            train_losses.append(df["train_loss"].min())
            computation_times.append(df["computation_time"].tolist())
            flops.append(
                df["computation_flops"].iloc[-1]
            )  # Assuming computation_flops is constant

            # Append baselines
            val_baselines.append(df["val_baseline"].mean())
            train_baselines.append(df["train_baseline"].mean())

            exp_param, exp_title, xaxis_title = experiment_parameter(
                exp_dir, key=exp_key
            )
            exp_parameters.append(exp_param)

    # Convert exp_parameters to numerical values for bar plot
    exp_numeric = range(len(exp_parameters))

    # Validation loss bar plot
    axes[0].bar(exp_numeric, val_losses, color="blue", label="Min Validation Loss")
    for baseline in val_baselines:
        axes[0].axhline(
            y=baseline, color="black", linestyle="--", label="Validation Baseline"
        )
    axes[0].set_xticks(exp_numeric)
    axes[0].set_xticklabels(exp_parameters, rotation=45, ha="right")
    axes[0].set_title("Min Validation Loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")

    # Training loss bar plot
    axes[1].bar(exp_numeric, train_losses, color="orange", label="Min Training Loss")
    for baseline in train_baselines:
        axes[1].axhline(
            y=baseline, color="black", linestyle="--", label="Training Baseline"
        )
    axes[1].set_xticks(exp_numeric)
    axes[1].set_xticklabels(exp_parameters, rotation=45, ha="right")
    axes[1].set_title("Min Training Loss")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")

    # Computation times boxplot
    axes[2].boxplot(computation_times)
    axes[2].set_xticklabels(exp_parameters, rotation=45, ha="right")
    axes[2].set_title("Computation Times")
    axes[2].set_ylabel("Time (seconds)")

    # FLOPs bar plot
    axes[3].bar(exp_numeric, flops, color="green", label="FLOPs")
    axes[3].set_xticks(exp_numeric)
    axes[3].set_xticklabels(exp_parameters, rotation=45, ha="right")
    axes[3].set_title("FLOPs")
    axes[3].set_ylabel("FLOPs")
    axes[3].set_yscale("log")

    # Layout adjustments
    plt.tight_layout()

    # Save or display the plot
    if exp_plot_dir:
        plt.savefig(os.path.join(exp_plot_dir, "experiment_summaries.png"), dpi=300)
        plt.close()
    else:
        plt.show()

    return fig, axes


def plot_loss_per_dataset(log_dir, mode="validation"):
    # Load train/validation losses
    losses = pd.read_csv(
        os.path.join(log_dir, "analysis", f"{mode}_loss_per_dataset.csv")
    )
    losses = losses.dropna().reset_index(drop=True)

    # Train dataset names
    train_info = pd.read_csv(os.path.join(log_dir, "dataset", "train_dataset_info.csv"))
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
    # ax.set_yscale("log")  # log scale
    ax.set_title(f"{mode.capitalize()} loss across datasets")
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
    exp_log_dir,
    exp_key,
    exp_plot_dir=None,
    mode="validation",
):
    # =============== Collect information ===============
    losses = pd.DataFrame(
        columns=["dataset", f"{mode}_loss", f"{mode}_baseline", "exp_param"]
    )

    # Loop through all experiments
    for file in np.sort(os.listdir(exp_log_dir)):
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
            os.path.join(exp_dir, "dataset", "train_dataset_info.csv")
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

    # Make exp_param multi index with dataset
    losses = losses.set_index(["exp_param", "dataset"])

    # Drop NaNs
    losses = losses.dropna().reset_index(drop=True)

    # Create one subplot per dataset, arranged in two columns
    num_datasets = len(losses.index.unique(level="dataset"))
    num_rows = int(np.ceil(num_datasets / 2))

    # =============== Start plotting ===============
    fig, ax = plt.subplots(num_rows, 2, figsize=(14, 12))
    sns.set_style("whitegrid")
    sns.set_palette("tab10")
    # Get a color palette with enough colors for all the datasets
    palette = sns.color_palette("tab10", len(losses.index.unique(level="dataset")))
    ax = ax.flatten()  # Flatten the ax array for easy iteration

    # Plot validation loss vs. exp_param (individual plots)
    for i, dataset in enumerate(losses.index.unique(level="dataset")):
        df_subset_model = losses.loc[
            losses.index.get_level_values("dataset") == dataset, f"{mode}_loss"
        ].reset_index()
        df_subset_baseline = losses.loc[
            losses.index.get_level_values("dataset") == dataset, f"{mode}_baseline"
        ].reset_index()

        sns.scatterplot(
            data=df_subset_model,
            x="exp_param",
            y=f"{mode}_loss",
            ax=ax[i],
            label="Model",
            marker="o",
        )
        sns.lineplot(
            data=df_subset_baseline,
            x="exp_param",
            y=f"{mode}_baseline",
            ax=ax[i],
            label="Baseline",
            linestyle="--",
            marker="o",
            color="black",
        )

        # Log-log scale
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")

        # Try to fit linear regression (log-log)
        try:
            x = np.log(df_subset_model["exp_param"].values)
            y = np.log(df_subset_model[f"{mode}_loss"].values)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            fit_label = (
                "y = {:.2e}x + {:.2e}\n".format(slope, intercept)
                + r"$R^2=$"
                + "{}".format(round(r_value**2, 4))
            )
            ax[i].plot(
                df_subset_model["exp_param"].values,
                np.exp(intercept + slope * x),
                color=palette[3],
                linestyle="-",
                label=fit_label,
            )
        except:
            logger.info(
                "Failed to fit linear regression (log-log scale) for dataset {}".format(
                    dataset
                )
            )
            pass

        # Add number of worms to title
        num_worms = losses.loc[
            losses.index.get_level_values("dataset") == dataset, "num_worms"
        ].values[0]
        ax[i].set_title(f"{dataset}: " + r"$n_{val}=$" + f"{int(num_worms)} worms")

        # Add text box with metadata
        model = losses.loc[
            losses.index.get_level_values("dataset") == dataset, "model_name"
        ].values[0]
        props = dict(boxstyle="round", facecolor="white", alpha=0.5)
        textstr = "Model: {}".format(model_name)
        ax[i].text(
            0.02,
            0.02,
            textstr,
            transform=ax[i].transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox=props,
        )

        # Only set x-label for bottom row
        if i >= len(ax) - 2:
            ax[i].set_xlabel(exp_xaxis)

        # Only set y-label for leftmost columns
        if i % 2 == 0:
            ax[i].set_ylabel("Loss")

        # Remove x and y labels for subplots that shouldn't have them
        if i < len(ax) - 2:
            ax[i].set_xlabel("")
        else:
            ax[i].set_xlabel(exp_xaxis)

        if i % 2 != 0:
            ax[i].set_ylabel("")

        ax[i].legend(loc="upper right")

    # Remove unused subplots
    if num_datasets % 2 != 0:
        ax[-1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(exp_plot_dir, f"{mode}_loss_per_dataset.png"), dpi=300)
    plt.close()

    # Plot loss vs. exp_param (comparison)
    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot loss vs. exp_param for all datasets
    for color_idx, dataset in enumerate(losses.index.unique(level="dataset")):
        df_subset_model = losses.loc[
            losses.index.get_level_values("dataset") == dataset, f"{mode}_loss"
        ].reset_index()
        df_subset_baseline = losses.loc[
            losses.index.get_level_values("dataset") == dataset, f"{mode}_baseline"
        ].reset_index()

        model_name = losses.loc[
            losses.index.get_level_values("dataset") == dataset, "model_name"
        ].values[0]

        color = palette[color_idx]

        sns.scatterplot(
            data=df_subset_model,
            x="exp_param",
            y=f"{mode}_loss",
            ax=ax,
            color=color,
            label=f"{model_name} (on {dataset})",
        )
        sns.lineplot(
            data=df_subset_baseline,
            x="exp_param",
            y=f"{mode}_baseline",
            ax=ax,
            linestyle="--",
            color=color,
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
            fit_label = f"y = {slope:.2f}x + {intercept:.2e} (R^2 = {r_value**2:.4f})"
            ax.plot(
                df_subset_model["exp_param"].values,
                np.exp(intercept + slope * x),
                linestyle="-",
                color=color,
                label=fit_label,
            )
        except:
            logger.info(
                "Failed to fit linear regression (log-log scale) for dataset {}".format(
                    dataset
                )
            )
            pass

    # Set axis labels and title
    ax.set_xlabel(exp_xaxis)
    ax.set_ylabel("Loss")
    ax.set_title(f"{mode.capitalize()} loss distributed across datasets")
    ax.legend(loc="upper right", fontsize="small")

    plt.tight_layout()

    if exp_plot_dir is not None:
        plt.savefig(
            os.path.join(exp_plot_dir, f"{mode}_loss_per_dataset_comparison.png"),
            dpi=300,
        )
        plt.close()
    else:
        return fig, ax
