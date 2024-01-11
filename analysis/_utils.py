from analysis._pkg import *


# Init logger
logger = logging.getLogger(__name__)


def hierarchical_clustering_algorithm(
    single_worm_data: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]],
    distance: str = "correlation",
    method: str = "ward",
    metric: Union[str, None] = None,
    truncate_mode: str = "lastp",
    p: int = 12,
    criterion: str = "maxclust",
    criterion_value: int = 4,
    save_fig: bool = False,
    verbose: bool = False,
    wormID: Union[str, None] = None,
) -> Tuple[np.ndarray, float]:
    """
    Perform hierarchical clustering on a single worm dataset.

    Args:
        single_worm_data (dict): A dictionary containing the single worm dataset.
        distance (str, optional): The distance metric to use. Defaults to "correlation".
        method (str, optional): The linkage method to use. Defaults to "ward".
        metric (str, optional): The distance metric to use. Defaults to None.
        truncate_mode (str, optional): The dendrogram truncation mode. Defaults to "lastp".
        p (int, optional): The number of clusters to display in the dendrogram. Defaults to 12.
        criterion (str, optional): The clustering criterion to use. Defaults to "maxclust".
        criterion_value (int, optional): The clustering criterion value to use. Defaults to 4.
        save_fig (bool, optional): Whether to save the dendrogram and distance matrix plots. Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        wormID (str, optional): The ID of the worm. Defaults to None.

    Returns:
        tuple: A tuple containing the computed clusters and the silhouette score.
    """

    np.set_printoptions(precision=4, suppress=True)

    if save_fig:
        plt.figure(figsize=(10, 3))
        plt.style.use("seaborn-whitegrid")

    X = single_worm_data["smooth_calcium_data"]  # (time, all neurons)
    X = X[:, single_worm_data["named_neurons_mask"]]  # (time, named and acive neurons)

    if distance == "correlation":
        R = np.corrcoef(X, rowvar=False)  # no correlated <- [0, 1] -> correlated
        R = (R + R.T) / 2  # Make it symmetric (just in case) -> numerical error
        D = 1 - R  # Distance matrix: close <- [0, 1] -> far
        np.fill_diagonal(D, 0)  # Make diagonal 0 (just in case)
        title_plot = "correlation matrix"

    elif distance == "cosine":
        R = X.T @ X  # Distance matrix: far <- [0, 1] -> close
        norms = np.linalg.norm(X, axis=0).reshape(-1, 1) @ np.linalg.norm(
            X, axis=0
        ).reshape(1, -1)
        R = (R / norms).detach().numpy()  # cosine similarity
        R = (R + R.T) / 2  # Make it symmetric (just in case) -> numerical error
        D = 1 - R  # Distance matrix: close <- [0, 1] -> far
        np.fill_diagonal(D, 0)  # Make diagonal 0 (just in case)
        title_plot = "cosine similarity matrix"

    if verbose:
        print("X.shape:", X.shape)
        print("Distance matrix shape:", D.shape)

    # The linkage function takes a condensed distance matrix, which is a flat array containing the upper triangular of the distance matrix.
    # We use squareform function to convert the matrix form to the condensed form.
    condensed_D = squareform(D)
    Z = linkage(condensed_D, method=method, metric=metric)

    # === Plot dendrogram ===

    dendrogram(
        Z,
        truncate_mode=truncate_mode,
        p=p,
        leaf_rotation=45.0,
        leaf_font_size=10.0,
        show_contracted=True,
    )
    if save_fig:
        plt.title("Hierarchical Clustering Dendrogram ({})".format(wormID))
        plt.xlabel("Cluster Size")
        plt.ylabel("Distance")
        plt.savefig("analysis/results/hierarchical_clustering/dendrogram.png", dpi=300)
        plt.close()

    # === Cluster labels ===
    computed_cluster_labels = fcluster(Z, criterion_value, criterion=criterion)
    silhouette_avg = sm.silhouette_score(
        D, computed_cluster_labels, metric="cosine"
    )  # Quality of the clustering -> cosine distance gave best results

    # === Sorting ===
    original_neuron_labels = np.array(
        [label for idx, label in single_worm_data["slot_to_named_neuron"].items()]
    )

    # Now we can sort the correlation matrix according to the cluster labels, and plot the correlation matrix again.
    sorted_R = R[:, np.argsort(computed_cluster_labels)]  # sort columns
    sorted_R = sorted_R[np.argsort(computed_cluster_labels), :]  # sort rows
    sorted_neuron_labels = original_neuron_labels[np.argsort(computed_cluster_labels)]
    sorted_computed_cluster_labels = computed_cluster_labels[
        np.argsort(computed_cluster_labels)
    ]

    if save_fig:
        plot_heat_map(
            R,
            title="Original {} ({})".format(title_plot, wormID),
            xlabel="Neuron",
            ylabel="Neuron",
            xticks=original_neuron_labels,
            yticks=original_neuron_labels,
            xtick_skip=2,
            ytick_skip=2,
        )
        plt.savefig(
            "analysis/results/hierarchical_clustering/original_distance_matrix.png",
            dpi=300,
        )
        plt.close()
        plot_heat_map(
            sorted_R,
            title="Sorted {} ({})".format(title_plot, wormID),
            xlabel="Neuron",
            ylabel="Neuron",
            xticks=sorted_neuron_labels,
            yticks=sorted_neuron_labels,
            xtick_skip=2,
            ytick_skip=2,
        )
        plt.savefig(
            "analysis/results/hierarchical_clustering/sorted_distance_matrix.png",
            dpi=300,
        )
        plt.close()

    # === Metrics ===
    file_path = "analysis/neuron_classification.json"

    try:
        with open(file_path, "r") as f:
            neuron_classification = json.load(f)
    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error while decoding JSON: {e}")

    clusters = {}
    for idx, neuron in enumerate(sorted_neuron_labels):
        clusters[neuron] = {
            "Computed Cluster": sorted_computed_cluster_labels[idx],
            "Reference": ", ".join(neuron_classification[neuron]),
        }

    clusters = pd.DataFrame.from_dict(clusters, orient="index")

    # Define the replacements
    replacements = {
        "interneuron": "I",
        "motor": "M",
        "sensory": "S",
        "motor, interneuron": "MI",
        "sensory, motor": "SM",
        "sensory, interneuron": "SI",
        "sensory, motor, interneuron": "SMI",
        "polymodal": "P",
    }

    # Replace the values in the 'Reference' column
    clusters["Reference"] = clusters["Reference"].replace(replacements)

    clusters.index.name = "Neuron"

    return clusters, silhouette_avg


def load_reference(group_by: Union[str, None] = None) -> Dict[str, str]:
    """
    Load neuron classification data from a JSON file and group neurons based on the specified criteria.

    Args:
        group_by (str, optional): The grouping criteria. Can be "four", "three", or None. Defaults to None.

    Returns:
        dict: A dictionary containing neuron IDs as keys and their corresponding group labels as values.
    """
    file_path = "analysis/neuron_classification.json"

    try:
        with open(file_path, "r") as f:
            neuron_classification = json.load(f)
    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error while decoding JSON: {e}")

    replacements = {
        "interneuron": "I",
        "motor": "M",
        "sensory": "S",
        "motor, interneuron": "MI",
        "sensory, motor": "SM",
        "sensory, interneuron": "SI",
        "sensory, motor, interneuron": "SMI",
        "unknown": "U",
    }

    for key, value in neuron_classification.items():
        text = ", ".join(neuron_classification[key])
        neuron_classification[key] = replacements[text]

    if group_by == "four":
        for key, value in neuron_classification.items():
            if value == "MI" or value == "SM" or value == "SI" or value == "SMI":
                neuron_classification[key] = "P"

            if value == "U":
                neuron_classification[key] = np.random.choice(["M", "I", "S"])

    elif group_by == "three":
        for key, value in neuron_classification.items():
            if value == "MI" or value == "SM" or value == "SI" or value == "SMI":
                neuron_classification[key] = np.random.choice([char for char in value])

            if value == "U":
                neuron_classification[key] = np.random.choice(["M", "I", "S"])

    elif group_by == None:
        for key, value in neuron_classification.items():
            if value == "U":
                neuron_classification[key] = np.random.choice(["M", "I", "S"])

    return neuron_classification


def loss_per_dataset(
    log_dir: str,
    experimental_datasets: dict,
    mode: Literal["train", "validation"] = "validation",
) -> None:
    """Uses the last model checkpoint in `log_dir` to evaluate the loss on
    either the train or validation set (depending on `mode`) of each dataset
    and set of worms in experimental_dataset. Saves the computed losses in a
    CSV file.

    Parameters
    ----------
    log_dir : str
        The directory where the model checkpoint and the dataset information are stored.
    experimental_datasets : dict
        A dictionary mapping the names of the experimental datasets to worms to select.
    mode : str, optional
        The mode to evaluate the loss. Either "train" or "validation". Default is "validation".

    Returns
    -------
    None
    """
    # Convert DictConfig to dict
    if isinstance(experimental_datasets, DictConfig):
        experimental_datasets = OmegaConf.to_object(experimental_datasets)

    logger.info(f"Analyzing {mode} loss per dataset...")

    # Retrieve information from training
    train_dataset_info = pd.read_csv(
        os.path.join(log_dir, "dataset", "train_dataset_info.csv"),
        converters={"neurons": ast.literal_eval},
    )

    seq_len = int(train_dataset_info["train_seq_len"].values[0])
    num_train_samples = int(train_dataset_info["num_train_samples"].values[0])
    train_split_ratio = float(train_dataset_info["train_split_ratio"].values[0])
    use_residual = bool(train_dataset_info["use_residual"].values[0])
    smooth_data = bool(train_dataset_info["smooth_data"].values[0])
    train_split_first = bool(train_dataset_info["train_split_first"].values[0])

    # Loss metrics
    running_base_loss = 0
    running_loss = 0

    dataset_names = []
    dataset_loss = []
    dataset_baseline = []
    num_worms = []

    # Load the model
    model_chkpt = os.path.join(log_dir, "train", "checkpoints", f"model_best.pt")
    model = get_model(OmegaConf.create({"use_this_pretrained_model": model_chkpt}))
    model.to(DEVICE)
    criterion = model.loss_fn()

    for dataset, worms_to_use in experimental_datasets.items():
        # Skip some datasets
        if worms_to_use is None:
            continue

        # Type check for `worms_to_use` to be int, list, or str
        assert isinstance(
            worms_to_use, (int, list, str)
        ), f"`worms_to_use` must be int, list, or str, but got {type(worms_to_use)}."

        # Create dataset
        combined_dataset, _ = create_combined_dataset(
            experimental_datasets={dataset: worms_to_use},
            num_named_neurons=None,  # use all available neurons
        )
        train_dataset, val_dataset, _ = split_combined_dataset(
            combined_dataset=combined_dataset,
            num_train_samples=num_train_samples,
            num_val_samples=num_train_samples,  # use the same number of samples as in the train dataset
            seq_len=seq_len,
            reverse=False,
            use_residual=use_residual,
            smooth_data=smooth_data,
            train_split_first=train_split_first,
            train_split_ratio=train_split_ratio,
        )
        select_dataset = train_dataset if mode == "train" else val_dataset

        num_worms.append(len(combined_dataset))

        # TODO: Make this analysis `batch_size` a configurable parameter
        batch_size = 32  # DEBUG
        dataloader = torch.utils.data.DataLoader(
            select_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        # Evaluation loop
        model.eval()

        with torch.no_grad():
            for batch_idx, (X, Y, mask, metadata) in enumerate(dataloader):
                X = X.to(DEVICE)
                Y = Y.to(DEVICE)
                mask = mask.to(DEVICE)

                # Baseline model is the naive predictor: predict that the value at
                # next time step is the same as the current value.
                y_base = X
                baseline = (
                    torch.tensor(0.0)
                    if model.version_2
                    else criterion(output=y_base, target=Y, mask=mask)
                )

                # All models operate sequence-to-sequence
                y_pred = model(X, mask)
                loss = criterion(output=y_pred, target=Y, mask=mask)

                # Update running losses
                running_base_loss += baseline.item()
                running_loss += loss.item()

            # Store metrics
            dataset_names.append(dataset)
            dataset_loss.append(running_loss / len(dataloader))
            dataset_baseline.append(running_base_loss / len(dataloader))

            # Reset running losses
            running_base_loss = 0
            running_loss = 0

    # Save losses in csv
    losses = pd.DataFrame(
        {
            "dataset": dataset_names,  # DEBUG
            f"{mode}_loss": dataset_loss,
            f"{mode}_baseline": dataset_baseline,
            "num_worms": num_worms,
        }
    )

    # Create analysis folder
    os.makedirs(os.path.join(log_dir, "analysis"), exist_ok=True)
    losses.to_csv(
        os.path.join(log_dir, "analysis", f"{mode}_loss_per_dataset.csv"),
        index=False,
    )
