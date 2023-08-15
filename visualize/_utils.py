from visualize._pkg import *

# Init logger
logger = logging.getLogger(__name__)

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


def plot_loss_curves(log_dir, info_to_display=None):
    """Plots the loss curves stored in a log directory.

    Args:
        log_dir (str): The path to the log directory.
    """

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.set(style='whitegrid')
    sns.set_palette("tab10")

    # Load loss curves
    loss_curves_csv = os.path.join(log_dir, 'train', 'train_metrics.csv')
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
    plt.legend(frameon=True, loc="upper right", fontsize=12)

    plt.title('Learning curves', fontsize=16)
    
    x_position_percent = 0.075  # Adjust this value to set the desired position
    x_position_box = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * x_position_percent
    y_position_percent = 0.80  # Adjust this value to set the desired position
    y_position_box = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * y_position_percent
    if info_to_display is not None:
        plt.text(x_position_box, y_position_box, info_to_display, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'), style='italic')
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(log_dir, 'train', 'loss_curves.png'), dpi=300)
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
        logger.error("No checkpoints found in the log directory.")
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


def plot_predictions(log_dir, neurons_to_plot=None, worms_to_plot=None):

    for type_pred in os.listdir(os.path.join(log_dir, 'prediction')):

        for type_ds in os.listdir(os.path.join(log_dir, 'prediction', type_pred)):
                
            for file in os.listdir(os.path.join(log_dir, 'prediction', type_pred, type_ds)):
            
                wormID = file[:-4]

                # Skip file if not .csv
                if not file.endswith('.csv') and wormID not in worms_to_plot:
                    continue

                url = os.path.join(log_dir, 'prediction', type_pred, type_ds, file)

                # Acess the prediction directory
                df = pd.read_csv(url)
                df.set_index(['Type', 'Unnamed: 1'], inplace=True)
                df.index.names = ['Type', '']

                # Load named neurons
                ds_info = pd.read_csv(os.path.join(log_dir, 'dataset', 'dataset_info.csv'))
                neurons = ds_info[ds_info['combined_dataset_index']==wormID]['neurons']
                neurons = ast.literal_eval(neurons.values[0]) # convert str to list

                seq_len = len(pd.concat([df.loc['Context'], df.loc['Ground Truth']], axis=0))
                time_vector = np.arange(seq_len)

                time_context = time_vector[:len(df.loc['Context'])]
                time_ground_truth = time_vector[len(df.loc['Context']):seq_len]
                time_generated = time_vector[len(df.loc['Context']):]

                sns.set_style('whitegrid')

                palette = sns.color_palette("tab10")
                ground_truth_color = palette[0]   # Red-like color
                generated_color = palette[3]   # Green-like color

                logger.info(f'Plotting neuron predictions for {type_pred}/{type_ds}/{wormID}...')

                for neuron in neurons:

                    # Skip neuron if not in neurons_to_plot
                    if neurons_to_plot is not None and neuron not in neurons_to_plot:
                        continue

                    fig, ax = plt.subplots(figsize=(10, 4))

                    ax.plot(time_context, df.loc['Context', neuron], color=ground_truth_color, label='Ground truth activity')
                    if not (type_pred == 'autoregressive'):
                        ax.plot(time_ground_truth, df.loc['Ground Truth', neuron], color=ground_truth_color, alpha=0.5)                    
                    ax.plot(time_generated, df.loc['Generated', neuron], color=generated_color, label='Model activity')

                    # Fill the context window
                    ax.axvspan(time_context[0], time_context[-1], alpha=0.1, color=ground_truth_color, label='Context window')

                    # Fill the generated window
                    ax.axvspan(time_generated[0], time_generated[-1], alpha=0.1, color=generated_color, label='Generation window')

                    ax.set_title(f'Neuronal Activity of {neuron}')
                    ax.set_xlabel('Time steps')
                    ax.set_ylabel('Activity ($\Delta F / F$)')
                    ax.legend(loc='upper right')

                    plt.tight_layout()

                    # Make figure directory
                    os.makedirs(os.path.join(log_dir, 'prediction', type_pred, type_ds, wormID), exist_ok=True)

                    plt.savefig(os.path.join(log_dir, 'prediction', type_pred, type_ds, wormID, f'{neuron}.png'), dpi=300)
                    plt.close()


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


def experiment_parameter(exp_dir, key):

    value, title, xaxis = None, None, None

    if key == 'default_multirun':
        value = exp_dir.split('/')[-1]
        title = 'Default multirun'
        xaxis = 'Experiment run'

    if key == 'num_time_steps':
        df = pd.read_csv(os.path.join(exp_dir, 'dataset', 'dataset_info.csv'))
        value = df['train_time_steps'].sum() # Total number of train time steps
        title = 'Amount of training data'
        xaxis = 'Number of time steps'

    if key == 'batch_size':
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, 'pipeline_info.yaml'))
        value = pipeline_info.submodule.train.batch_size # Experiment batch size
        title = 'Batch size'
        xaxis = 'Batch size'

    if key == 'lr':
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, 'pipeline_info.yaml'))
        value = pipeline_info.submodule.train.lr # Learning rate used for training
        title = 'Learning rate'
        xaxis = 'Learning rate'

    if key == 'num_named_neurons':
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, 'pipeline_info.yaml'))
        value = pipeline_info.submodule.dataset.num_named_neurons # Number of named neurons used for training
        title = 'Neuron population'
        xaxis = 'Number of neurons'

    if key == 'seq_len':
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, 'pipeline_info.yaml'))
        value = pipeline_info.submodule.dataset.seq_len # Sequence length used for training
        title = 'Sequence length'
        xaxis = 'Sequence length'

    if key == 'num_train_samples':
        pipeline_info = OmegaConf.load(os.path.join(exp_dir, 'pipeline_info.yaml'))
        value = pipeline_info.submodule.dataset.num_train_samples # Number of training samples used for training
        title = 'Number of training samples'
        xaxis = 'Number of training samples'
    
    if key == 'computation_time':
        df = pd.read_csv(os.path.join(exp_dir, 'train', 'train_metrics.csv'))
        value = (df['train_computation_time'].min(), df['train_computation_time'].mean(), df['train_computation_time'].max()) # Computation time
        title = 'Computation time'
        xaxis = 'Computation time (s)'
    
    return value, title, xaxis


def plot_exp_losses(exp_log_dir, exp_plot_dir, exp_name):
    ''' 
        * Plot validation loss curves and baselines for all experiments
        * Plot computation time for all experiments
    '''

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    sns.set_style('whitegrid')

    # Store computation time and parameters
    computation_time = []
    parameters = []

    # Loop over all the experiment files
    for file in np.sort(os.listdir(exp_log_dir)):

        # Skip if not starts with exp
        if not file.startswith('exp') or file.startswith('exp_'):
            continue

        # Get experiment directory
        exp_dir = os.path.join(exp_log_dir, file)
        
        # Load train metrics
        df = pd.read_csv(os.path.join(exp_dir, 'train', 'train_metrics.csv'))

        # Experiment parameters
        exp_param, exp_title, exp_xaxis = experiment_parameter(exp_dir, key=exp_name)
        ct_param, ct_title, ct_xaxis = experiment_parameter(exp_dir, key='computation_time')

        # Store computation time and parameters
        computation_time.append(ct_param)
        parameters.append(exp_param)

        # Plot validation loss
        ax[0].plot(df['epoch'], df['val_loss'], label=exp_param)
        # Plot validation baseline
        ax[0].plot(df['epoch'], df['val_baseline'], color='black', linestyle='--')

    # Set loss labels
    ax[0].set_xlabel('Epoch', fontsize=12)
    ax[0].set_ylabel('Validation loss', fontsize=12)

    # Set loss legend
    legend = ax[0].legend(fontsize=10)
    legend.set_title(exp_xaxis)

    # Set loss title
    ax[0].set_title(exp_title + ' experiment', fontsize=14)

    # Plot computation time with error bars
    y = np.array(computation_time)[:,1].T # mean
    yerr = np.array(computation_time)[:,::2].T # min max variation
    yerr[0,:] = y - yerr[0,:]
    yerr[1,:] = yerr[1,:] - y
    ax[1].errorbar(parameters, y, yerr=yerr, fmt='o', capsize=2, ecolor='grey', color='black', elinewidth=1)

    # Regression line computation time
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(parameters, y)
        x = np.linspace(np.min(parameters), np.max(parameters), 100)
        ct_reg_label = 'y = {:.2e}x + {:.2e}'.format(slope, intercept)
        ax[1].plot(x, intercept + slope*x, color='red', linestyle='-.', alpha=0.5, label=ct_reg_label)
    except:
        pass

    # Set computation time labels
    ax[1].set_xlabel(exp_xaxis, fontsize=12)
    ax[1].set_ylabel('Computation time (s)', fontsize=12)

    # Set computation time title
    ax[1].set_title(ct_title, fontsize=14)

    # Set computation time legend
    legend = ax[1].legend(fontsize=10)

    plt.tight_layout()

    # Save figure
    fig.savefig(os.path.join(exp_plot_dir, 'val_loss.png'))
    plt.close()