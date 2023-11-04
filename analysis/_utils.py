from analysis._pkg import *


# Init logger
logger = logging.getLogger(__name__)


def hierarchical_clustering_algorithm(
    single_worm_data,
    distance="correlation",
    method="ward",
    metric=None,
    truncate_mode="lastp",
    p=12,
    criterion="maxclust",
    criterion_value=4,
    save_fig=False,
    verbose=False,
    wormID=None,
):
    """
    single_worm_data: single worm dataset
    method: linkage method
    metric: distance metric
    plot: whether to plot the dendrogram or not
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


def load_reference(group_by=None):
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


def validation_loss_per_dataset(log_dir, experimental_datasets):
    logger.info("Analyzing validation loss per dataset...")

    # Retrieve information from training
    train_dataset_info = pd.read_csv(
        os.path.join(log_dir, "dataset", "train_dataset_info.csv")
    )
    seq_len = int(train_dataset_info["train_seq_len"].values[0])
    num_train_samples = int(train_dataset_info["num_train_samples"].values[0])
    use_residual = int(train_dataset_info["use_residual"].values[0])
    smooth_data = int(train_dataset_info["smooth_data"].values[0])

    # Loss metrics
    val_running_base_loss = 0
    val_running_loss = 0

    dataset_val_loss = []
    dataset_val_baseline = []
    num_worms = []

    # Load the model
    model_chkpt = os.path.join(log_dir, "train", "checkpoints", f"model_best.pt")
    model = get_model(OmegaConf.create({"use_this_pretrained_model": model_chkpt}))
    model.to(DEVICE)
    criterion = model.loss_fn()

    for dataset, worms_to_use in experimental_datasets.items():
        # Skip some datasets
        if worms_to_use is None:
            dataset_val_loss.append(np.NaN)
            dataset_val_baseline.append(np.NaN)
            num_worms.append(np.NaN)
            continue

        combined_dataset, _ = create_combined_dataset(
            experimental_datasets={dataset: worms_to_use}, num_named_neurons="all"
        )
        _, val_dataset, _ = split_combined_dataset(
            combined_dataset=combined_dataset,
            num_train_samples=num_train_samples,
            num_val_samples=num_train_samples,  # use the same number of samples as in the train dataset
            seq_len=seq_len,
            use_residual=use_residual,
            smooth_data=smooth_data,
            reverse=False,
        )

        num_worms.append(len(combined_dataset))

        valloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,  # TODO: Get the batch size from config
        )

        # Evaluation loop
        model.eval()

        with torch.no_grad():
            for batch_idx, (X_val, Y_val, masks_val, metadata_val) in enumerate(
                valloader
            ):
                X_val = X_val.to(DEVICE)
                Y_val = Y_val.to(DEVICE)
                masks_val = masks_val.to(DEVICE)

                # Baseline model: identity model - predict that the next time step is the same as the current one.
                # This is the simplest model we can think of: predict that the next time step is the same as the current one
                # is better than predict any other random number.
                y_base = X_val
                val_baseline = compute_loss_vectorized(
                    loss_fn=criterion, X=y_base, Y=Y_val, masks=masks_val
                )

                # All models operate sequence-to-sequence
                y_pred = model(X_val, masks_val)
                val_loss = compute_loss_vectorized(
                    loss_fn=criterion, X=y_pred, Y=Y_val, masks=masks_val
                )

                # Update running losses
                val_running_base_loss += val_baseline.item()
                val_running_loss += val_loss.item()

            # Store metrics
            dataset_val_loss.append(val_running_loss / len(valloader))
            dataset_val_baseline.append(val_running_base_loss / len(valloader))

            # Reset running losses
            val_running_base_loss = 0
            val_running_loss = 0

    # Save losses in csv
    losses = pd.DataFrame(
        {
            "dataset": list(experimental_datasets.keys()),
            "val_loss": dataset_val_loss,
            "val_baseline": dataset_val_baseline,
            "num_worms": num_worms,
        }
    )

    # Create analysis folder
    os.makedirs(os.path.join(log_dir, "analysis"), exist_ok=True)
    losses.to_csv(
        os.path.join(log_dir, "analysis", "validation_loss_per_dataset.csv"),
        index=False,
    )
