from analysis._pkg import *


def find_config_files(root_dir):
    for file in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file)
        if file == "config.yaml":
            yield file_path
        elif os.path.isdir(file_path) and not file.startswith("."):
            for config_file in find_config_files(file_path):
                yield config_file


def plot_loss_vs_parameter(config_dir, varied_param, control_param, subplot_param):
    """
    Plots the minimum validation loss against a varied parameter for different levels of a control parameter.
    Creates a separate subplot for each unique value of a third subplot parameter.

    Args:
        config_dir (str): Directory containing the config files.
        varied_param (str): Parameter that is varied across the experiments.
        control_param (str): Parameter that controls the color of the plot lines.
        subplot_param (str): Parameter that determines the creation of separate subplots.

    Returns:
        df (pandas.DataFrame): DataFrame containing the data used to create the plot.
    """
    # Find the config files and their corresponding loss values
    configs = {}  # dict maps config file path to (loss, config)
    for file_path in find_config_files(config_dir):
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
            parent_dir = os.path.dirname(file_path)
            if os.path.exists(os.path.join(parent_dir, "loss_curves.csv")):
                loss_df = pd.read_csv(
                    os.path.join(parent_dir, "loss_curves.csv"), index_col=0
                )
                loss = loss_df["centered_test_losses"][
                    loss_df["centered_test_losses"].idxmin()
                ]
                configs[os.path.dirname(file_path)] = (loss, OmegaConf.create(data))

    # Split the parameters into their components
    varied_param = varied_param.split(".")
    control_param = control_param.split(".")
    subplot_param = subplot_param.split(".")

    # Create a data frame with the relevant data
    records = []
    for cfg_path, loss_cfg_tuple in configs.items():
        val = loss_cfg_tuple[1][varied_param[0]][varied_param[1]]
        lvl = loss_cfg_tuple[1][control_param[0]][control_param[1]]
        sub = loss_cfg_tuple[1][subplot_param[0]][subplot_param[1]]
        loss = loss_cfg_tuple[0]
        records.append((val, lvl, sub, loss))

    df = pd.DataFrame(
        records,
        columns=[
            ".".join(varied_param),
            ".".join(control_param),
            ".".join(subplot_param),
            "loss",
        ],
    )

    # Sort the DataFrame by the varied parameter
    df = df.sort_values(".".join(varied_param))

    # Count the unique values in subplot_param to determine the number of rows (subplots)
    num_subplots = df[".".join(subplot_param)].nunique()

    # Define a base subplot height (you can adjust this as needed)
    base_subplot_height = 2.5

    # If there is only one subplot, increase the height
    if num_subplots == 1:
        subplot_height = 4
    else:
        # Otherwise, use the base height
        subplot_height = base_subplot_height

    # Create a grid of subplots, with one row for each unique value of 'subplot_param'
    # The grid will share the y-axis across all subplots
    g = sns.FacetGrid(
        df, row=".".join(subplot_param), height=subplot_height, aspect=4, sharey=True
    )

    # For each subplot, plot a lineplot of 'loss' against 'varied_param',
    # with different colors for each unique value of 'control_param'
    # 'errorbar="sd"' and 'err_style="band"' to display standard deviation as shaded areas around the mean
    g.map(
        sns.lineplot,
        ".".join(varied_param),
        "loss",
        ".".join(control_param),
        err_style="band",
        errorbar="sd",
    )

    # Customize the axis labels and title for each subplot
    g.set_axis_labels("%s %s" % (varied_param[0], varied_param[1]), "Validation Loss")
    g.fig.subplots_adjust(top=0.95)  # Adjust the Figure in `g`, increased to 0.95
    g.fig.suptitle(
        "Scaling plot: loss vs {} {} \n Validation loss after training on different {} {}".format(
            *varied_param, *control_param
        ),
        fontsize="large",
        y=1.11,  # Adjust y to push the title upwards a little
    )

    # Add a legend to the figure
    g.add_legend(title="{} {}".format(*control_param))

    # Set the y-axis limits for all subplots
    g.set(ylim=(-0.1, None))

    # Set x-axis scale to log if the varied parameter is train_size or hidden_size
    if varied_param[1] in {"worm_timesteps", "hidden_size"}:
        g.set(xscale="log")

    # Save the figure as an image
    if not os.path.exists("figures"):
        os.mkdir("figures")
    g.savefig("figures/scaling_plot_val_loss_vs_{}_{}.png".format(*varied_param))

    # Return the DataFrame
    return df


def hierarchical_clustering_algorithm(dataset_names, distance='correlation',
                                     method='ward', metric=None,
                                     truncate_mode='lastp', p=12,
                                     criterion='maxclust', criterion_value=4, verbose=False,
                                     show_plots=True):
    """
        single_worm_data: single worm dataset
        method: linkage method
        metric: distance metric
        plot: whether to plot the dendrogram or not
    """

    # Load data

    one_dataset = np.random.choice(dataset_names)
    dataset_names = {'dataset': {
        'name': str(one_dataset),
        }
    }
    dataset_config = OmegaConf.create(dataset_names)
    dataset = get_dataset(dataset_config) # load random dataset
    wormid = np.random.choice([key for key in dataset.keys()]) # pick random worm
    single_worm_data = dataset[wormid]

    np.set_printoptions(precision=4, suppress=True)
    if show_plots:
        plt.figure(figsize=(10, 3))
        plt.style.use('seaborn-whitegrid')

    X = single_worm_data['smooth_calcium_data'] # (time, all neurons)
    X = X[:, single_worm_data['named_neurons_mask']]  # (time, named and acive neurons)

    if distance == 'correlation':
        R = np.corrcoef(X, rowvar=False) # no correlated <- [0, 1] -> correlated
        R = (R + R.T) / 2  # Make it symmetric (just in case) -> numerical error
        D = 1 - R # Distance matrix: close <- [0, 1] -> far
        np.fill_diagonal(D, 0) # Make diagonal 0 (just in case)
        title_plot = "correlation matrix"
        
    elif distance == 'cosine':
        R = X.T @ X # Distance matrix: far <- [0, 1] -> close
        norms = np.linalg.norm(X, axis=0).reshape(-1,1) @ np.linalg.norm(X, axis=0).reshape(1,-1)
        R = (R / norms).detach().numpy() # cosine similarity
        R = (R + R.T) / 2  # Make it symmetric (just in case) -> numerical error
        D = 1 - R # Distance matrix: close <- [0, 1] -> far
        np.fill_diagonal(D, 0) # Make diagonal 0 (just in case)
        title_plot = "cosine similarity matrix"

    elif distance == 'dtw':
        X = X.detach().numpy().astype(np.double).T
        time_series = [vec for vec in X]
        R = dtw.distance_matrix_fast(time_series, window=1000) # window: anti-phase dynamics around 50s and 100s
        D = R
        title_plot = "DTW matrix"

    if verbose:
            print("X.shape:", X.shape)
            print("Distance matrix shape:", D.shape)

    # The linkage function takes a condensed distance matrix, which is a flat array containing the upper triangular of the distance matrix. 
    # We use squareform function to convert the matrix form to the condensed form.
    condensed_D = squareform(D)
    Z = linkage(condensed_D, method=method, metric=metric)

    # === Plot dendrogram ===
    if show_plots:
        dendrogram(Z, truncate_mode=truncate_mode, p=p, leaf_rotation=45., leaf_font_size=10., show_contracted=True)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Cluster Size')
        plt.ylabel('Distance')
        plt.show()

    # === Cluster labels ===
    computed_cluster_labels = fcluster(Z, criterion_value, criterion=criterion)
    silhouette_avg = sm.silhouette_score(D, computed_cluster_labels, metric='cosine') # Quality of the clustering -> cosine distance gave best results


    # === Sorting ===
    original_neuron_labels = np.array([label for idx, label in single_worm_data['slot_to_named_neuron'].items()])

    # Now we can sort the correlation matrix according to the cluster labels, and plot the correlation matrix again.
    sorted_R = R[:, np.argsort(computed_cluster_labels)] # sort columns
    sorted_R = sorted_R[np.argsort(computed_cluster_labels), :] # sort rows
    sorted_neuron_labels = original_neuron_labels[np.argsort(computed_cluster_labels)]
    sorted_computed_cluster_labels = computed_cluster_labels[np.argsort(computed_cluster_labels)]

    if show_plots:
        plot_heat_map(R, title="Original " + title_plot, xlabel="Neuron", ylabel="Neuron", xticks=original_neuron_labels, yticks=original_neuron_labels, xtick_skip=2, ytick_skip=2)
        plot_heat_map(sorted_R, title="Sorted " + title_plot, xlabel="Neuron", ylabel="Neuron", xticks=sorted_neuron_labels, yticks=sorted_neuron_labels, xtick_skip=2, ytick_skip=2)

    # === Metrics ===
    file_path = 'analysis/neuron_classification.json'

    try:
        with open(file_path, 'r') as f:
            neuron_classification = json.load(f)
    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error while decoding JSON: {e}")

    clusters = {}
    for idx, neuron in enumerate(sorted_neuron_labels):
        clusters[neuron] = {'Computed Cluster': sorted_computed_cluster_labels[idx], 'Reference': ', '.join(neuron_classification[neuron])}

    clusters = pd.DataFrame.from_dict(clusters, orient='index')

    # Define the replacements
    replacements = {
        'interneuron': 'I',
        'motor': 'M',
        'sensory': 'S',
        'motor, interneuron': 'MI',
        'sensory, motor': 'SM',
        'sensory, interneuron': 'SI',
        'sensory, motor, interneuron': 'SMI',
        'polymodal': 'P'
    }

    # Replace the values in the 'Reference' column
    clusters['Reference'] = clusters['Reference'].replace(replacements)

    clusters.index.name = 'Neuron'

    return clusters, silhouette_avg