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


def neuron_distribution(
    df,
    ref_dict,
    stat="percent",
    group_by=None,
    plot_type="both",
    save_fig=False,
    wormID=None,
):
    assert group_by in [
        None,
        "three",
        "four",
    ], f"Invalid group_by: {group_by} -> Must be None, 'three' or 'four'"

    assert stat in [
        "percent",
        "count",
        "proportion",
        "density",
    ], f"Invalid stat: {stat} -> Must be 'percent', 'count', 'proportion' or 'density'"

    new_df = df.copy()

    if group_by == "four":
        # assert just 4 unique keys in ref_dict
        assert (
            len(set(ref_dict.values())) == 4
        ), f"Invalid ref_dict -> Must have 4 unique values"
    elif group_by == "three":
        # assert just 3 unique keys in ref_dict
        assert (
            len(set(ref_dict.values())) == 3
        ), f"Invalid ref_dict -> Must have 3 unique values"
    else:
        # assert just 8 unique keys in ref_dict
        assert (
            len(set(ref_dict.values())) == 7
        ), f"Invalid ref_dict -> Must have 7 unique values"

    # Replace all references by the ref_dict ones (Group the neurons by three or four)
    for neuron in new_df.index:
        new_df.loc[neuron, "Reference"] = ref_dict[neuron]

    # Create a figure with the desired number of subplots
    if plot_type == "both":
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    elif plot_type == "ground-truth":
        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        axes = [axes]
    elif plot_type == "computed-cluster":
        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        axes = [None, axes]

    if plot_type == "both" or plot_type == "ground-truth":
        # Create the histogram (literature)
        sns.histplot(
            data=new_df, x="Reference", stat=stat, discrete=True, kde=True, ax=axes[0]
        )

        # Set the labels and title for the first subplot
        axes[0].set_title("Ground truth labels distribution ({})".format(wormID))
        axes[0].set_xlabel("Neuron type")

    if plot_type == "both" or plot_type == "computed-cluster":
        # Create the histogram (computed clusters)
        hist = sns.histplot(
            data=new_df,
            x="Computed Cluster",
            stat=stat,
            discrete=True,
            kde=True,
            ax=axes[1],
        )

        # Set the labels and title for the second subplot
        axes[1].set_title("Computed cluster labels distribution ({})".format(wormID))
        axes[1].set_xlabel("Neuron type")

        # Change the xticks to the correct labels
        axes[1].set_xticks(np.arange(len(set(new_df["Computed Cluster"]))) + 1)

    # Compute the proportions of each Reference label within each bin
    if stat == "percent" and (plot_type == "both" or plot_type == "computed-cluster"):
        color_palette = sns.color_palette("Set1")
        unique_references = new_df["Reference"].unique()[
            : len(set(new_df["Computed Cluster"]))
        ]
        color_map = {
            ref: color_palette[i % len(color_palette)]
            for i, ref in enumerate(unique_references)
        }

        for patch in hist.patches:
            x = patch.get_x()
            width = patch.get_width()
            height = patch.get_height()
            bin_label = int(x + width / 2)  # Compute the label of the bin
            proportions = new_df[new_df["Computed Cluster"] == bin_label][
                "Reference"
            ].value_counts(normalize=True)
            cumulative_height = 0
            for ref, proportion in proportions.items():
                color = color_map.get(ref, "gray")
                ref_height = height * proportion
                axes[1].bar(
                    x + width / 2,
                    ref_height,
                    width=width,
                    bottom=cumulative_height,
                    color=color,
                    label=ref,
                    alpha=0.5,
                    edgecolor="black",
                )
                cumulative_height += ref_height

        # Add legend for the first four items
        legend_elements = [
            Patch(facecolor=color_map.get(ref, "gray"), edgecolor="black", label=ref)
            for ref in new_df["Reference"].unique()[
                : len(set(new_df["Computed Cluster"]))
            ]
        ]
        axes[1].legend(handles=legend_elements, loc="upper right")

        # Adjust the layout and spacing between subplots
        plt.tight_layout()

        # Save plot
        if save_fig:
            plt.savefig(
                "analysis/results/hierarchical_clustering/cluster_distribution.png",
                dpi=300,
            )
        plt.close()

    return new_df


def create_total(df):
    new_df = df.copy()
    new_df.loc["total"] = new_df.sum(axis=0)  # Count over columns
    new_df["total"] = new_df.sum(axis=1)  # Count over rows
    return new_df


def delete_total(df):
    new_df = df.copy()
    new_df = new_df.drop("total", axis=0)  # Drop row
    new_df = new_df.drop("total", axis=1)  # Drop column
    return new_df


def count_inside_clusters(df, percentage=False, dimension="reference"):
    new_df = df.copy()

    new_df = (
        new_df.groupby("Computed Cluster")["Reference"]
        .value_counts()
        .unstack()
        .fillna(0)
    )

    new_df = new_df.astype(int)
    new_df = create_total(new_df)

    if percentage:
        new_df = convert_to_percentages(new_df, dimension=dimension)

    return new_df


def convert_to_percentages(df, dimension="reference"):
    assert dimension in [
        "reference",
        "computed-cluster",
    ], f"Invalid dimension: {dimension} -> Must be 'reference' or 'computed-cluster'"

    new_df = df.copy()

    # create total row and column if they don't exist
    if "total" not in new_df.index:
        new_df = create_total(new_df)

    if dimension == "reference":
        new_df = new_df.div(new_df.loc["total"], axis=1) * 100
    elif dimension == "computed-cluster":
        new_df = new_df.div(new_df["total"], axis=0) * 100

    return new_df.round(decimals=2)


def suggest_classification(computed_clusters_df):
    # TODO: suggestion 2

    new_df = computed_clusters_df.copy()

    # Index of the max values per columns
    count_df = delete_total(
        count_inside_clusters(new_df, percentage=True, dimension="reference")
    )
    max_values_col = count_df.idxmax(axis=0)

    # Column of the max value per row
    count_df = delete_total(
        count_inside_clusters(new_df, percentage=True, dimension="computed-cluster")
    )
    max_values_row = count_df.idxmax(axis=1)

    # Create mapping
    suggestion = {
        "hip1": {key: value for key, value in max_values_row.items()},
        "hip2": {},
    }  # hip1: inside cluster, hip2: global

    return suggestion


def cluster2suggestion(value, suggestion):
    return suggestion[value]


def create_ref_column(df, ref_dict):
    new_df = df.copy()
    new_df["Reference"] = [
        ref_dict[neuron] if neuron in ref_dict.keys() else np.NaN for neuron in df.index
    ]
    return new_df


def delete_ref_column(df):
    new_df = df.copy()
    new_df = new_df.drop("Reference", axis=1)
    return new_df


def hc_analyse_dataset(
    dataset_names,
    apply_suggestion=False,
    hip="hip1",
    group_by="four",
    method="ward",
    metric=None,
):
    """
    dataset = loaded dataset
    """

    # Check wether we are using synthetic data
    for name in dataset_names:
        if "synthetic" in name.lower():
            synthetic = True
            break
        else:
            synthetic = False

    # Load data
    dataset_config = OmegaConf.create({"dataset": {"name": dataset_names}})
    dataset = get_datasets(dataset_config)  # load dataset

    if group_by == "four":
        groups = 4
        num_clusters = 4
    elif group_by == "three":
        groups = 3
        num_clusters = 3
    elif group_by == "None":
        groups = 7
        num_clusters = 7
        group_by = None  # convert 'None' to None (yaml file)

    ref_dict = load_reference(group_by=group_by)  # Create same ref dict for all worms

    num_worms = len(dataset.keys())

    if not apply_suggestion:
        print("No suggestion applied, ignoring hip parameter.\n")
    else:
        print(f"Suggestion applied: {hip}.")

    # ===

    silhouettes = []
    all_worm_clusters_list = []

    # random pick a worm ID for plotting
    random_wormID = np.random.choice(list(dataset.keys()))

    # ===

    for i, wormID in enumerate(dataset.keys()):
        if len(dataset[wormID]["named_neuron_to_slot"]) == 0:
            continue  # Skip worm if no neurons are found

        if wormID == random_wormID:
            save_fig = True
        else:
            save_fig = False

        clusters, silhouette_avg = hierarchical_clustering_algorithm(
            dataset[wormID],
            method=method,
            metric=metric,
            save_fig=save_fig,
            criterion="maxclust",
            criterion_value=num_clusters,
            wormID=wormID,
        )  # Compute clusters

        silhouettes.append(silhouette_avg)  # Save silhouette score

        grouped_clusters = neuron_distribution(
            clusters,
            ref_dict=ref_dict,
            group_by=group_by,
            save_fig=save_fig,
            wormID=wormID,
        )  # Group clusters

        sugg_dict = suggest_classification(grouped_clusters)  # Suggest classification

        if apply_suggestion:
            all_worm_clusters_list.append(
                grouped_clusters["Computed Cluster"]
                .apply(cluster2suggestion, suggestion=sugg_dict[hip])
                .drop(columns=["Reference"])
            )
        else:
            all_worm_clusters_list.append(
                grouped_clusters["Computed Cluster"].drop(columns=["Reference"])
            )

        if i % 10 == 0:
            print(f"{i}/{num_worms} worms analysed")

    all_worm_clusters = pd.concat(
        all_worm_clusters_list, axis=1, keys=range(1, len(all_worm_clusters_list) + 1)
    )
    all_worm_clusters.columns = [
        f"worm{i}" for i in range(0, len(all_worm_clusters_list))
    ]

    all_worm_clusters = create_ref_column(
        all_worm_clusters, ref_dict
    )  # Add reference column

    if apply_suggestion and not synthetic:
        # Accuracy of the classification for each worm
        for wormID in all_worm_clusters.columns[:-1]:
            # Select the wormN and reference columns
            s = all_worm_clusters[[wormID, "Reference"]].copy().dropna()
            # Count +1 for each match between the wormN and reference columns
            count = s.apply(
                lambda x: 1 if x[wormID] == x["Reference"] else 0, axis=1
            ).sum()
            # Create row for the accuracy of the worm
            all_worm_clusters.loc["accuracy", wormID] = count / len(s)

        # Accuracy of the classification for each neuron
        for neuron in all_worm_clusters.index[:-1]:
            # Compare the classifications of the neuron and compare to its reference
            s = all_worm_clusters.loc[neuron].iloc[:-1].copy().dropna().value_counts()
            ref = all_worm_clusters.loc[neuron, "Reference"]
            # Create row for the accuracy of the neuron if not NaN
            if (s.get(ref, 0) / s.sum()) != np.NaN:
                # if (s[ref] / s.sum()) != np.NaN:
                all_worm_clusters.loc[neuron, "accuracy"] = s[ref] / s.sum()
            else:
                all_worm_clusters.loc[neuron, "accuracy"] = 0.0

        all_worm_clusters = delete_ref_column(
            all_worm_clusters
        )  # Delete reference column
        all_worm_clusters = create_ref_column(
            all_worm_clusters, ref_dict
        )  # Add reference column

        # Average accuracy across all individuals
        print(
            f"Average acc across individuals: {np.mean(all_worm_clusters.loc['accuracy']).round(4)}"
        )
        # Averace accuracy for each neuron
        print(
            f"Average acc across neurons: {np.mean(all_worm_clusters['accuracy']).round(4)}"
        )

    all_worm_clusters.to_csv(
        "analysis/results/hierarchical_clustering/worm_clusters.csv", index=True
    )

    # === Plot silhouettes ===

    # Create a DataFrame from the silhouette averages
    s_data = {"Silhouette Averages": silhouettes}
    s_df = pd.DataFrame(s_data)

    # Create the box plot using Seaborn
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=s_df,
        y="Silhouette Averages",
        boxprops={"facecolor": "steelblue", "alpha": 0.6},
        capprops={"color": "black"},
        whiskerprops={"color": "black"},
        medianprops={"color": "black"},
    )

    # Set the plot labels
    plt.xlabel("Silhouette Averages")
    plt.ylabel("")
    plt.title("Silhouette Averages")
    plt.tight_layout()
    # Write the median value
    plt.text(
        0.33,
        np.median(silhouettes) + 0.003,
        f"{np.median(silhouettes).round(4)}",
        color="black",
    )
    # Write the max value
    plt.text(
        0.14,
        np.max(silhouettes) + 0.003,
        f"{np.max(silhouettes).round(4)}",
        color="black",
    )
    # Write the min value
    plt.text(
        0.14,
        np.min(silhouettes) + 0.003,
        f"{np.min(silhouettes).round(4)}",
        color="black",
    )

    # Show the plot
    plt.savefig(
        "analysis/results/hierarchical_clustering/silhouette_averages.png", dpi=300
    )
    plt.close()

    return all_worm_clusters, ref_dict, silhouettes


def validation_loss_per_dataset(log_dir, experimental_datasets, task):
    logger.info("Analyzing validation loss per dataset...")

    # Retrieve information from training
    train_dataset_info = pd.read_csv(
        os.path.join(log_dir, "dataset", "train_dataset_info.csv")
    )
    seq_len = int(train_dataset_info["train_seq_len"].values[0])
    num_train_samples = int(train_dataset_info["num_train_samples"].values[0])
    k_splits = int(train_dataset_info["k_splits"].values[0])
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
            k_splits=k_splits,
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

                # If many-to-one prediction, select last time step. Else, many-to-many prediction.
                if task == "many-to-one":
                    y_base = X_val[:, -1, :].unsqueeze(1)  # Select last time step
                    Y_val = Y_val[:, -1, :].unsqueeze(1)  # Select last time step
                else:
                    y_base = X_val

                # Baseline model: identity model - predict that the next time step is the same as the current one.
                # This is the simplest model we can think of: predict that the next time step is the same as the current one
                # is better than predict any other random number.
                val_baseline = compute_loss_vectorized(
                    loss_fn=criterion, X=y_base, Y=Y_val, masks=masks_val
                )

                # Model
                y_pred = model(X_val, masks_val)

                if task == "many-to-one":
                    y_pred = y_pred[:, -1, :].unsqueeze(1)  # Select last time step

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
