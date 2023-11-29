import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import stats
from scipy.stats import linregress, t
from visualize._utils import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde


# 1. Marker and Dataset color codes
def legend_code():
    markers = {
        "o": "LSTM",
        "s": "Transformer",
        "^": "Feedforward",
    }

    model_labels = {
        "NetworkLSTM": "LSTM",
        "NeuralTransformer": "Transformer",
        "FeatureFFNN": "Feedforward",
    }

    marker_colors = sns.color_palette("tab10", n_colors=len(markers))

    # Create custom markers for models
    marker_legend = [
        Line2D([0], [0], marker=m, color=marker_colors[i], label=l, linestyle="None")
        for i, (m, l) in enumerate(markers.items())
    ]

    # Plot the marker legends
    fig, axs = plt.subplots(1, 2, figsize=(4, 1))

    # Plot marker legend on the left subplot
    axs[0].legend(handles=marker_legend, loc="center", title="Model")
    # Legend title italic
    axs[0].get_legend().get_title().set_fontsize("large")
    axs[0].get_legend().get_title().set_fontstyle("italic")
    axs[0].axis("off")

    color_palette = sns.color_palette("tab10", n_colors=8)
    # Add black color to the end of color palette
    color_palette.append((0, 0, 0))

    datasets = [
        "Kato",
        "Nichols",
        "Skora",
        "Kaplan",
        "Yemini",
        "Uzel",
        "Flavell",
        "Leifer",
    ]

    dataset_labels = [
        "Kato (2015)",
        "Nichols (2017)",
        "Skora (2018)",
        "Kaplan (2020)",
        "Yemini (2021)",
        "Uzel (2022)",
        "Flavell (2023)",
        "Leifer (2023)",
    ]
    original_dataset_names = [
        "Kato2015",
        "Nichols2017",
        "Skora2018",
        "Kaplan2020",
        "Yemini2021",
        "Uzel2022",
        "Flavell2023",
        "Leifer2023",
    ]

    # Create rectangular color patches for datasets
    color_legend = [
        Patch(facecolor=c, edgecolor=c, label=l)
        for c, d, l in zip(color_palette, datasets, dataset_labels)
    ]

    # Plot color legend on the right subplot
    axs[1].legend(handles=color_legend, loc="center", title="Experimental datasets")
    axs[1].get_legend().get_title().set_fontsize("large")
    axs[1].get_legend().get_title().set_fontstyle("italic")
    axs[1].axis("off")

    ds_color_code = {dataset: color for dataset, color in zip(datasets, color_palette)}
    original_ds_color_code = {
        dataset: color for dataset, color in zip(original_dataset_names, color_palette)
    }
    model_marker_code = {
        model: marker for model, marker in zip(markers.values(), markers.keys())
    }
    model_color_code = {
        model: color for model, color in zip(markers.values(), marker_colors)
    }

    plt.show()

    leg_code = {
        "ds_color_code": ds_color_code,
        "original_ds_color_code": original_ds_color_code,
        "model_marker_code": model_marker_code,
        "model_color_code": model_color_code,
        "color_legend": color_legend,
        "dataset_labels": dataset_labels,
        "marker_colors": marker_colors,
        "marker_legend": marker_legend,
        "model_labels": model_labels,
    }

    return leg_code


def dataset_information(path_dict, legend_code):
    """
    path_dict: dictionary with the path to train_dataset_info,
        validation_dataset_info and combined_dataset_info.
    """

    # ### LOAD IN DATASET INFORMATION ###
    train_dataset_info = pd.read_csv(path_dict["train_dataset_info"])
    val_dataset_info = pd.read_csv(path_dict["val_dataset_info"])
    combined_dataset_info = pd.read_csv(path_dict["combined_dataset_info"])

    train_dataset_info["total_time_steps"] = (
        train_dataset_info["train_time_steps"] + val_dataset_info["val_time_steps"]
    )
    train_dataset_info["tsn"] = (
        train_dataset_info["total_time_steps"] / train_dataset_info["num_neurons"]
    )
    amount_of_data_distribution = (
        train_dataset_info[["dataset", "total_time_steps"]]
        .groupby("dataset")
        .sum()
        .sort_values(by="total_time_steps", ascending=False)
    )
    amount_of_data_distribution["percentage"] = (
        amount_of_data_distribution["total_time_steps"]
        / amount_of_data_distribution["total_time_steps"].sum()
    )

    # ########### SET UP FOR FIGURES ###########
    # Get color code and legend from legend_code
    ds_color_code = legend_code["ds_color_code"]
    color_legend = legend_code["color_legend"]

    # Initialize figure
    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 0.25])

    # Assigning the subplots to positions in the grid
    ax1 = plt.subplot(gs[0, 0])  # Top left, 'Number of worms analyzed' pie chart
    ax2 = plt.subplot(gs[0, 1])  # Top right, 'Number of neurons per worm' bar plot
    ax3 = plt.subplot(
        gs[1, 0]
    )  # Bottom left, 'Total duration of recorded neural activity' pie chart
    ax4 = plt.subplot(
        gs[1, 1]
    )  # Bottom middle, 'Duration of recorded neural activity per worm' bar plot
    ax5 = plt.subplot(gs[0, 3])  # Bottom right, legend
    ax6 = plt.subplot(gs[0, 2])
    ax7 = plt.subplot(gs[1, 2:4])

    # ########### FOR WORMS PIE CHART ###############
    num_worms_per_dataset = combined_dataset_info[["dataset", "original_index"]]
    # Count the unique 'original_index' for each 'dataset'
    num_worms_per_dataset = (
        num_worms_per_dataset.groupby("dataset")["original_index"]
        .nunique()
        .reset_index(name="num_worms")
    )
    # Calculate the percentage for each dataset
    num_worms_per_dataset["percentage"] = (
        num_worms_per_dataset["num_worms"] / num_worms_per_dataset["num_worms"].sum()
    )
    # Sort the values by percentage in descending order
    num_worms_per_dataset = num_worms_per_dataset.sort_values(
        by="percentage", ascending=False
    )
    worm_count_label = num_worms_per_dataset["num_worms"][:7].tolist() + [""]
    # Plotting the worms per dataset pie chart
    ax1.pie(
        num_worms_per_dataset["num_worms"],
        labels=[
            f"{percentage:.1%}" for percentage in num_worms_per_dataset["percentage"]
        ],
        labeldistance=1.075,
        startangle=45,
        colors=[
            ds_color_code[dataset[:-4]] for dataset in num_worms_per_dataset["dataset"]
        ],
    )
    ax1.pie(
        num_worms_per_dataset["num_worms"],
        labels=[f"{n}" for n in worm_count_label],
        labeldistance=0.70,
        startangle=45,
        colors=[
            ds_color_code[dataset[:-4]] for dataset in num_worms_per_dataset["dataset"]
        ],
    )
    ax1.set_title("(A) Number of worms in dataset", fontsize=14)

    # ########### NEURON POPULATION DISTRIBUTION BAR PLOT ###############
    ax2 = plt.subplot(gs[0, 1])  # Subplot for 'Number of neurons per worm' bar plot
    # Compute data for the neuron population distribution bar plot
    neuron_pop_stats = (
        train_dataset_info.groupby("dataset")["num_neurons"]
        .agg(["mean", "sem"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )
    neuron_pop_colors = neuron_pop_stats["dataset"].apply(
        lambda x: ds_color_code.get(x.split("20")[0], "grey")
    )
    ax2.bar(
        neuron_pop_stats["dataset"],
        neuron_pop_stats["mean"],
        yerr=2 * neuron_pop_stats["sem"],
        color=neuron_pop_colors,
        capsize=5,
    )
    # Add a dashed horizontal line at y=302
    ax2.axhline(y=302, color="black", linestyle="dashed", linewidth=1, alpha=0.5)
    # Annotate the line
    # Use the right edge of the subplot for text annotation to prevent overflow
    right_edge = ax2.get_xlim()[1]
    ax2.text(
        right_edge,  # x position at the right edge
        302,  # y position at the line
        "Number of neurons in C. elegans hermaphrodite",
        horizontalalignment="right",  # Align text to the right
        fontsize=10,
        fontstyle="italic",
    )
    ax2.set_title("(B) Number of recorded neurons per worm", fontsize=14)
    ax2.set_ylabel("Neuron population size")
    ax2.set_xticklabels(neuron_pop_stats["dataset"], rotation=45, ha="right")
    ax2.set_xticks([])  # Delete xticks

    # ########### TOTAL DURATION OF RECORDED NEURAL ACTIVITY PIE CHART ###############
    ax3 = plt.subplot(
        gs[1, 0]
    )  # Subplot for 'Total duration of recorded neural activity' pie chart
    # Compute data for total duration pie chart
    total_duration_stats = (
        train_dataset_info.groupby("dataset")["total_time_steps"]
        .sum()
        .reset_index()
        .sort_values(by="total_time_steps", ascending=False)
    )
    total_duration_stats["percentage"] = (
        total_duration_stats["total_time_steps"]
        / total_duration_stats["total_time_steps"].sum()
    )

    # Determine the smallest slice
    smallest_slice_index = total_duration_stats["percentage"].idxmin()

    # Generate labels, omitting the smallest slice
    labels = [
        f"{percentage:.1%}" if i != smallest_slice_index else ""
        for i, percentage in enumerate(total_duration_stats["percentage"])
    ]

    total_duration_colors = total_duration_stats["dataset"].apply(
        lambda x: ds_color_code.get(x.split("20")[0], "grey")
    )
    # Plotting the total duration pie chart
    ax3.pie(
        total_duration_stats["total_time_steps"],
        labels=labels,
        labeldistance=1.075,
        startangle=90,
        colors=total_duration_colors,
    )
    ax3.set_title("(C) Total duration of recorded neural activity", fontsize=14)

    # ########### DURATION OF RECORDED NEURAL ACTIVITY PER WORM BAR PLOT ###############
    ax4 = plt.subplot(
        gs[1, 1]
    )  # Subplot for 'Duration of recorded neural activity per worm' bar plot
    # Compute data for recording duration bar plot
    recording_duration_stats = (
        train_dataset_info.groupby("dataset")["total_time_steps"]
        .agg(["mean", "sem"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )
    recording_duration_colors = recording_duration_stats["dataset"].apply(
        lambda x: ds_color_code.get(x.split("20")[0], "grey")
    )
    ax4.bar(
        recording_duration_stats["dataset"],
        recording_duration_stats["mean"],
        yerr=2 * recording_duration_stats["sem"],
        color=recording_duration_colors,
        capsize=5,
    )
    # Add a dashed horizontal line at y=3600
    ax4.axhline(y=3600, color="black", linestyle="dashed", linewidth=1, alpha=0.5)
    # Annotate the line
    # Use the right edge of the subplot for text annotation to prevent overflow
    right_edge = ax4.get_xlim()[1]
    ax4.text(
        right_edge,  # x position at the right edge
        3600,  # y position at the line
        "3600 seconds = 1 hour of calcium imaging",
        horizontalalignment="right",  # Align text to the right
        fontsize=10,
        fontstyle="italic",
    )
    ax4.set_title("(D) Duration of recorded neural activity per worm", fontsize=14)
    ax4.set_ylabel("Time (s)")
    ax4.set_xticklabels(recording_duration_stats["dataset"], rotation=45, ha="right")
    ax4.set_xticks([])  # Delete xticks

    # ########### TIME STEPS PER NEURON BAR PLOT ###############
    ax6 = plt.subplot(
        gs[0, 2]
    )  # Subplot for 'Number of time steps per recorded neuron' bar plot
    # Compute data for time steps per neuron bar plot
    ts_per_neuron_stats = (
        train_dataset_info.groupby("dataset")["tsn"]
        .agg(["mean", "sem"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )
    ts_per_neuron_colors = ts_per_neuron_stats["dataset"].apply(
        lambda x: ds_color_code.get(x.split("20")[0], "grey")
    )
    ax6.bar(
        ts_per_neuron_stats["dataset"],
        ts_per_neuron_stats["mean"],
        yerr=2 * ts_per_neuron_stats["sem"],
        color=ts_per_neuron_colors,
        capsize=5,
    )
    # Add a dashed horizontal line at y=100
    ax6.axhline(y=100, color="black", linestyle="dashed", linewidth=1, alpha=0.5)
    # Annotate the line
    # Use the right edge of the subplot for text annotation to prevent overflow
    right_edge = ax6.get_xlim()[1]
    ax6.text(
        right_edge,  # x position at the right edge
        100,  # y position at the line
        "Sequence length $L=100$ was used in our experiments",
        horizontalalignment="right",  # Align text to the right
        fontsize=10,
        fontstyle="italic",
    )
    ax6.set_title("(E) Number of time steps per recorded neuron", fontsize=14)
    ax6.set_ylabel("Time steps per neuron")
    ax6.set_xticklabels(ts_per_neuron_stats["dataset"], rotation=45, ha="right")
    ax6.set_xticks([])  # Delete xticks

    # ########### SAMPLING INTERVAL BAR PLOT ###############
    ax7 = plt.subplot(
        gs[1, 2:4]
    )  # Subplot for 'Sampling interval of recorded neural activity' bar plot
    # Compute data for the sampling interval bar plot
    dt_stats = (
        train_dataset_info.groupby("dataset")["original_median_dt"]
        .agg(["mean", "sem"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )
    dt_colors = dt_stats["dataset"].apply(
        lambda x: ds_color_code.get(x.split("20")[0], "grey")
    )
    ax7.bar(
        dt_stats["dataset"],
        dt_stats["mean"],
        yerr=2 * dt_stats["sem"],
        color=dt_colors,
        capsize=5,
    )
    # Add a dashed horizontal line at y=1.0
    ax7.axhline(y=1.0, color="black", linestyle="dashed", linewidth=1, alpha=0.5)
    # Annotate the line
    # Use the right edge of the subplot for text annotation to prevent overflow
    right_edge = ax7.get_xlim()[1]
    ax7.text(
        right_edge,  # x position at the right edge
        1.0,  # y position at the line
        "We downsampled all data to $\Delta s = 1.0s$ (1 Hz)",
        horizontalalignment="right",  # Align text to the right
        fontsize=10,
        fontstyle="italic",
    )
    ax7.set_title("(F) Sampling interval of recorded neural activity", fontsize=14)
    ax7.set_ylabel(r"Mean sampling interval ($\Delta$s)")
    ax7.set_xticklabels(dt_stats["dataset"], rotation=45, ha="right")
    ax7.set_xticks([])  # Delete xticks

    # ########### LEGEND SUBPLOT ###############
    ax5 = plt.subplot(gs[0, 3])  # Subplot for legend
    ax5.legend(
        handles=color_legend,
        loc="center",
        title="Experimental datasets",
        fontsize=11,
        title_fontsize=12,
    )
    ax5.get_legend().get_title().set_fontstyle("italic")
    ax5.axis("off")

    plt.tight_layout()
    plt.show()

    # Construct and return the dataset_info dictionary with all the computed stats
    dataset_info = {
        "train_dataset_info": train_dataset_info,
        "amount_of_data_distribution": amount_of_data_distribution,
        "num_worms_per_dataset": num_worms_per_dataset,
        "total_duration_stats": total_duration_stats,
        "neuron_pop_stats": neuron_pop_stats,
        "recording_duration_stats": recording_duration_stats,
        "ts_per_neuron_stats": ts_per_neuron_stats,
        "dt_stats": dt_stats,
    }

    return dataset_info


# 3a. Create dataframe with data and hidden scaling results for plotting
def data_scaling_df(nts_experiments):
    """
    nts_experiments: dictionary with the paths to the experiments
        - num_time_steps: The total number of train time steps
        - time_steps_per_neuron: The average number of train time steps per neuron
        - num_named_neurons: The number of distinct named neurons recorded across all worms
        - num_train_samples: The number of training sequences sampled per worm
        - hidden_size: The hidden size of the model
        - batch_size: The batch size used for training
        - seq_len: The sequence length used for training
        - learn_rate: The learning rate used for training
        - dataset: The name of a single dataset used for training
        - model: The type of neural net model used for training
        - optimizer: The type of optimizer used for training
        - time_last_epoch: The computation time in seconds for the last epoch
        - computation_flops: The number of floating point operations (FLOPs)
        - num_parameters: The total number of trainable parameters in the model
    """
    data_scaling_results = {
        "experiment_ID": [],
        "model_type": [],
        "hidden_size": [],
        "num_parameters": [],
        "num_worms": [],
        "num_time_steps": [],
        "time_steps_per_neuron": [],
        "num_named_neurons": [],
        "min_val_loss": [],
        "val_baseline": [],
    }

    for model_name, exp_paths in nts_experiments.items():
        for exp_log_dir in exp_paths:
            # Loop over all the experiment files
            for exp_ID in np.sort(os.listdir(exp_log_dir)):
                # Skip if not starts with exp
                if not exp_ID.startswith("exp") or exp_ID.startswith("exp_"):
                    continue

                exp_dir = os.path.join(exp_log_dir, exp_ID)

                # Load train metrics
                df = pd.read_csv(os.path.join(exp_dir, "train", "train_metrics.csv"))

                data_scaling_results["experiment_ID"].append(exp_ID)
                data_scaling_results["model_type"].append(
                    experiment_parameter(exp_dir, "model_type")[0]
                )
                data_scaling_results["hidden_size"].append(
                    experiment_parameter(exp_dir, "hidden_size")[0]
                )
                data_scaling_results["num_parameters"].append(
                    experiment_parameter(exp_dir, "num_parameters")[0]
                )
                data_scaling_results["num_worms"].append(
                    experiment_parameter(exp_dir, "num_worms")[0]
                )
                data_scaling_results["num_time_steps"].append(
                    experiment_parameter(exp_dir, "num_time_steps")[0]
                )
                data_scaling_results["time_steps_per_neuron"].append(
                    experiment_parameter(exp_dir, "time_steps_per_neuron")[0]
                )
                data_scaling_results["num_named_neurons"].append(
                    experiment_parameter(exp_dir, "num_named_neurons")[0]
                )
                data_scaling_results["min_val_loss"].append(df["val_loss"].min())
                data_scaling_results["val_baseline"].append(df["val_baseline"].median())

    return pd.DataFrame(data_scaling_results)


# 3b. Data scaling plot
def data_scaling_plot(data_scaling_df, legend_code):
    model_marker_code = legend_code["model_marker_code"]
    model_color_code = legend_code["model_color_code"]
    marker_legend = legend_code["marker_legend"]
    model_labels = legend_code["model_labels"]

    # Group by model_label and expID, and compute the mean and std
    data_scaling_results = data_scaling_df

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_xscale("log")
    ax.set_yscale("log")

    model_names = data_scaling_results["model_type"].unique()

    # Plot
    for model_idx, model in enumerate(model_names):
        nts = data_scaling_results[data_scaling_results["model_type"] == model][
            "num_time_steps"
        ].values
        val_loss = data_scaling_results[data_scaling_results["model_type"] == model][
            "min_val_loss"
        ].values
        baseline_loss = data_scaling_results[
            data_scaling_results["model_type"] == model
        ]["val_baseline"].values

        model_label = model_labels[model]

        ax.scatter(
            nts,
            val_loss,
            marker=model_marker_code[model_label],
            color=model_color_code[model_label],
            alpha=0.5,
        )

        if model_idx < 1:
            # Plot horizontal line for baseline
            ax.plot(
                np.linspace(np.min(nts), np.max(nts), 10000),
                np.ones(10000) * baseline_loss[0],
                label="Baseline",
                color="black",
                alpha=0.5,
                linestyle="--",
            )

        # Regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.log(nts), np.log(val_loss)
        )
        fit_label = (
            "y = {:.2f}x + {:.1f}\n".format(slope, intercept)
            + r"$R^2=$"
            + "{:.3f}".format(r_value**2)
        )
        x = np.linspace(np.min(nts), np.max(nts), 10000)
        ax.plot(
            x,
            np.exp(intercept + slope * np.log(x)),
            color=model_color_code[model_label],
            label=fit_label,
        )

    # Handles and labels for first legend
    handles, labels = ax.get_legend_handles_labels()

    # Create the first legend
    legend1 = ax.legend(
        handles=marker_legend,
        loc="center right",
        bbox_to_anchor=(0.995, 0.85),
        title="Model architecture",
    )
    ax.get_legend().get_title().set_fontstyle("italic")
    ax.get_legend().get_title().set_fontsize("large")
    ax.add_artist(legend1)

    ax.legend(
        handles, labels, loc="center left", bbox_to_anchor=(0.01, 0.25), fontsize=12
    )

    ax.set_xlabel("Num. train time steps", fontsize=14, fontweight="bold")
    ax.set_ylabel("Validation MSE Loss", fontsize=14, fontweight="bold")

    plt.show()


# 3c. Hidden dimension scaling (actually num. time steps scaling)
def hidden_scaling_plot(data_scaling_df, legend_code):
    model_marker_code = legend_code["model_marker_code"]
    model_color_code = legend_code["model_color_code"]
    marker_legend = legend_code["marker_legend"]
    model_labels = legend_code["model_labels"]

    # Group by model_label and expID, and compute the mean and std
    data_scaling_results = data_scaling_df

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xscale("log")
    ax.set_yscale("log")

    model_names = data_scaling_results["model_type"].unique()

    # Plot
    for model_idx, model in enumerate(model_names):
        hdv_mean = data_scaling_results[data_scaling_results["model_type"] == model][
            "num_parameters"
        ].values

        val_loss_mean = data_scaling_results[
            data_scaling_results["model_type"] == model
        ]["min_val_loss"].values

        baseline_mean = data_scaling_results[
            data_scaling_results["model_type"] == model
        ]["val_baseline"].values

        model_label = model_labels[model]
        ax.scatter(
            hdv_mean,
            val_loss_mean,
            marker=model_marker_code[model_label],
            color=model_color_code[model_label],
        )

        if model_idx == 0:
            ax.plot(
                np.sort(hdv_mean),
                np.sort(baseline_mean),
                label="Baseline",
                color="black",
                alpha=0.5,
                linestyle="--",
            )

    # Handles and labels for first legend
    handles, labels = ax.get_legend_handles_labels()

    # Create the first legend
    legend1 = ax.legend(
        handles=marker_legend,
        loc="center right",
        bbox_to_anchor=(0.995, 0.15),  # 0.995, 0.15
        title="Model",
    )
    ax.get_legend().get_title().set_fontstyle("italic")
    ax.get_legend().get_title().set_fontsize("large")
    ax.add_artist(legend1)

    ax.legend(
        handles, labels, loc="center left", bbox_to_anchor=(0.85, 0.95), fontsize=10
    )  # 0.001, 0.76

    ax.set_xlabel(
        "Num. trainable parameters",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Validation MSE Loss", fontsize=14, fontweight="bold")

    plt.show()


# 4a. Scaling slopes by model by individual exp. dataset results for plotting
def scaling_slopes_df(nts_experiments):
    data_scaling_results = {
        "experiment_ID": [],
        "model_type": [],
        "validation_dataset": [],
        "individual_validation_loss": [],
        "individual_baseline_loss": [],
        "num_parameters": [],
        "num_worms": [],
        "num_time_steps": [],
        "min_val_loss": [],
        "val_baseline": [],
    }

    for model_label, exp_paths in nts_experiments.items():
        for exp_log_dir in exp_paths:
            # Loop over all the experiment files
            for experiment_ID in sorted(
                os.listdir(exp_log_dir), key=lambda x: x.strip("exp_")
            ):
                # Skip if not starts with exp
                if not experiment_ID.startswith("exp") or experiment_ID.startswith(
                    "exp_"
                ):
                    continue

                exp_dir = os.path.join(exp_log_dir, experiment_ID)

                # Load train metrics
                df = pd.read_csv(os.path.join(exp_dir, "train", "train_metrics.csv"))
                df_analysis = pd.read_csv(
                    os.path.join(exp_dir, "analysis", "validation_loss_per_dataset.csv")
                )

                for val_dataset in df_analysis["dataset"]:
                    data_scaling_results["validation_dataset"].append(val_dataset)
                    data_scaling_results["individual_validation_loss"].append(
                        df_analysis[df_analysis["dataset"] == val_dataset][
                            "validation_loss"
                        ].values[0]
                    )
                    data_scaling_results["individual_baseline_loss"].append(
                        df_analysis[df_analysis["dataset"] == val_dataset][
                            "validation_baseline"
                        ].values[0]
                    )
                    data_scaling_results["experiment_ID"].append(experiment_ID)
                    data_scaling_results["model_type"].append(
                        experiment_parameter(exp_dir, "model_type")[0]
                    )
                    data_scaling_results["num_parameters"].append(
                        experiment_parameter(exp_dir, "num_parameters")[0]
                    )
                    data_scaling_results["num_worms"].append(
                        experiment_parameter(exp_dir, "num_worms")[0]
                    )
                    data_scaling_results["num_time_steps"].append(
                        experiment_parameter(exp_dir, "num_time_steps")[0]
                    )
                    data_scaling_results["min_val_loss"].append(df["val_loss"].min())
                    data_scaling_results["val_baseline"].append(
                        df["val_baseline"].median()
                    )

    return pd.DataFrame(data_scaling_results)


# 4b. Scaling slopes by model by individual exp. dataset plot
def scaling_slopes_plot(scaling_slope_results, legend_code):
    """
    Plot the scaling slopes for different models and validation datasets.

    Args:
        scaling_slope_results (DataFrame): DataFrame containing the scaling slope results.
        legend_code (dict): Dictionary containing the legend codes for colors and markers.

    Returns:
        None
    """

    # Extract necessary information from legend_code
    original_ds_color_code = legend_code["original_ds_color_code"]
    model_marker_code = legend_code["model_marker_code"]
    model_type_to_label = legend_code["model_labels"]

    # Create a figure with three subplots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Initialize a dictionary to store the slopes
    slopes = {
        "model_type": [],
        "validation_dataset": [],
        "slope": [],
    }

    # Iterate over unique model types
    model_labels = []
    for subplot_idx, model_type in enumerate(
        scaling_slope_results["model_type"].unique()
    ):
        model_labels.append(model_type_to_label[model_type])

        # Iterate over unique validation datasets
        for val_dataset in scaling_slope_results["validation_dataset"].unique():
            # Filter the data for the current model and validation dataset
            filtered_results = scaling_slope_results[
                (scaling_slope_results["model_type"] == model_type)
                & (scaling_slope_results["validation_dataset"] == val_dataset)
            ]

            x = filtered_results["num_time_steps"].values
            y = filtered_results["individual_validation_loss"].values

            # Compute the baseline values
            baseline = filtered_results["individual_baseline_loss"].values
            baseline_mean = np.mean(baseline)

            # Plot the baseline
            ax[subplot_idx].plot(
                np.linspace(min(x), max(x), len(baseline)),
                baseline,
                linestyle="--",
                color=original_ds_color_code[val_dataset],
            )

            # Plot scatter plots of  each individual dataset
            ax[subplot_idx].scatter(
                x, y, s=5, alpha=0.1, color=original_ds_color_code[val_dataset]
            )

            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.log(x), np.log(y)
            )

            # Create a range of x-values for plotting the regression line and error band
            x_range = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
            y_range = np.exp(intercept + slope * np.log(x_range))

            # Estimate the density with a Gaussian kernel
            kde = gaussian_kde(np.log(y), bw_method="silverman")
            y_std = kde(np.log(y_range)) * std_err  # Multiplying by std_err for scaling

            # Plot the regression line
            ax[subplot_idx].plot(
                x_range, y_range, color=original_ds_color_code[val_dataset], zorder=10
            )

            # Plot the error bands around the regression line
            ax[subplot_idx].fill_between(
                x_range,
                y_range - y_std,
                y_range + y_std,
                color=original_ds_color_code[val_dataset],
                alpha=0.3,
                zorder=5,
            )

            # Set axes to use a log-log scale
            ax[subplot_idx].set_xscale("log")
            ax[subplot_idx].set_yscale("log")

            # Store the slope information
            slopes["model_type"].append(model_type)
            slopes["validation_dataset"].append(val_dataset)
            slopes["slope"].append(slope)

    # Create a DataFrame from the slopes dictionary
    slopes = pd.DataFrame(slopes)

    # Set axis labels and title for each subplot
    ax[0].set_xlabel(
        "Num. train time steps", fontdict={"fontsize": 14, "fontweight": "bold"}
    )
    ax[1].set_xlabel(
        "Num. train time steps", fontdict={"fontsize": 14, "fontweight": "bold"}
    )
    ax[2].set_xlabel(
        "Num. train time steps", fontdict={"fontsize": 14, "fontweight": "bold"}
    )
    ax[0].set_ylabel(
        "Validation MSE Loss", fontdict={"fontsize": 14, "fontweight": "bold"}
    )
    ax[0].set_title(f"(A) {model_labels[0]} model", fontdict={"fontsize": 16})
    ax[1].set_title(f"(B) {model_labels[1]} model", fontdict={"fontsize": 16})
    ax[2].set_title(f"(C) {model_labels[2]} model", fontdict={"fontsize": 16})

    # Display the legend in the top right subplot
    ax[2].legend(
        handles=legend_code["color_legend"],
        loc="center right",
        bbox_to_anchor=(1.5, 0.5),
        title="Experimental datasets",
    )

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()


############################################################


def compute_confidence_interval(x, y, confidence=0.95):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    y_fit = slope * x + intercept
    mean_x = np.mean(x)
    n = len(x)
    t_val = t.ppf((1 + confidence) / 2.0, n - 2)
    conf_int = (
        t_val * std_err * np.sqrt(1 / n + (x - mean_x) ** 2 / np.sum((x - mean_x) ** 2))
    )
    return y_fit, conf_int


# 4b. Scaling slopes by model by individual exp. dataset plot
def scaling_slopes_plot(scaling_slope_results, legend_code):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    model_labels = []

    for subplot_idx, model_type in enumerate(
        scaling_slope_results["model_type"].unique()
    ):
        model_labels.append(legend_code["model_labels"][model_type])
        ax_model = ax[subplot_idx]
        model_data = scaling_slope_results[
            scaling_slope_results["model_type"] == model_type
        ]

        for val_dataset in model_data["validation_dataset"].unique():
            dataset_data = model_data[model_data["validation_dataset"] == val_dataset]
            x = np.log(dataset_data["num_time_steps"].values)
            y = np.log(dataset_data["individual_validation_loss"].values)
            y_fit, conf_int = compute_confidence_interval(x, y)

            # Plot the regression line
            ax_model.plot(
                dataset_data["num_time_steps"],
                np.exp(y_fit),
                label=val_dataset,
                color=legend_code["original_ds_color_code"][val_dataset],
            )

            # Plot the confidence interval
            ax_model.fill_between(
                dataset_data["num_time_steps"],
                np.exp(y_fit - 2 * conf_int),
                np.exp(y_fit + 2 * conf_int),
                color=legend_code["original_ds_color_code"][val_dataset],
                alpha=0.3,
            )

        ax_model.set_xscale("log")
        ax_model.set_yscale("log")
        ax_model.set_title(f"{model_labels[subplot_idx]}")
        ax_model.set_xlabel("Num. train time steps")
        if subplot_idx == 0:
            ax_model.set_ylabel("Validation MSE Loss")

        # Plot baselines
        for val_dataset in model_data["validation_dataset"].unique():
            baseline = model_data[model_data["validation_dataset"] == val_dataset][
                "individual_baseline_loss"
            ].mean()
            ax_model.axhline(
                baseline,
                ls="--",
                color=legend_code["original_ds_color_code"][val_dataset],
            )

    handles, labels = ax[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", title="Experimental datasets")
    plt.tight_layout()
    plt.show()


############################################################

# # 4b. Scaling slopes by model by individual exp. dataset plot
# def scaling_slopes_plot(scaling_slope_results, legend_code):
#     """
#     Plot the scaling slopes for different models and validation datasets.

#     Args:
#         scaling_slope_results (DataFrame): DataFrame containing the scaling slope results.
#         legend_code (dict): Dictionary containing the legend codes for colors and markers.

#     Returns:
#         None
#     """

#     # Extract necessary information from legend_code
#     original_ds_color_code = legend_code["original_ds_color_code"]
#     model_marker_code = legend_code["model_marker_code"]
#     model_type_to_label = legend_code["model_labels"]

#     # Create a figure with three subplots
#     fig, ax = plt.subplots(1, 3, figsize=(15, 5))

#     # Initialize a dictionary to store the slopes
#     slopes = {
#         "model_type": [],
#         "validation_dataset": [],
#         "slope": [],
#     }

#     # Iterate over unique model types
#     model_labels = []
#     for subplot_idx, model_type in enumerate(
#         scaling_slope_results["model_type"].unique()
#     ):
#         model_labels.append(model_type_to_label[model_type])

#         # Iterate over unique validation datasets
#         for val_dataset in scaling_slope_results["validation_dataset"].unique():
#             # Filter the data for the current model and validation dataset
#             filtered_results = scaling_slope_results[
#                 (scaling_slope_results["model_type"] == model_type)
#                 & (scaling_slope_results["validation_dataset"] == val_dataset)
#             ]

#             x = filtered_results["num_time_steps"].values
#             y = filtered_results["individual_validation_loss"].values

#             # Compute the baseline values
#             baseline = filtered_results["individual_baseline_loss"].values
#             baseline_mean = np.mean(baseline)

#             # Plot the baseline
#             ax[subplot_idx].plot(
#                 np.linspace(min(x), max(x), len(baseline)),
#                 baseline,
#                 linestyle="--",
#                 color=original_ds_color_code[val_dataset],
#             )

#             # Plot scatter plots of  each individual dataset
#             ax[subplot_idx].scatter(
#                 x, y, s=7, alpha=0.4, color=original_ds_color_code[val_dataset]
#             )

#             # Perform linear regression
#             slope, intercept, r_value, p_value, std_err = stats.linregress(
#                 np.log(x), np.log(y)
#             )

#             # Create a range of x-values for plotting the regression line and error band
#             x_range = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
#             y_range = np.exp(intercept + slope * np.log(x_range))

#             # Plot the regression line
#             ax[subplot_idx].plot(
#                 x_range, y_range, color=original_ds_color_code[val_dataset], zorder=10
#             )

#             # Set axes to use a log-log scale
#             ax[subplot_idx].set_xscale("log")
#             ax[subplot_idx].set_yscale("log")

#             # Store the slope information
#             slopes["model_type"].append(model_type)
#             slopes["validation_dataset"].append(val_dataset)
#             slopes["slope"].append(slope)

#     # Create a DataFrame from the slopes dictionary
#     slopes = pd.DataFrame(slopes)

#     # Set axis labels and title for each subplot
#     ax[0].set_xlabel(
#         "Num. train time steps", fontdict={"fontsize": 14, "fontweight": "bold"}
#     )
#     ax[1].set_xlabel(
#         "Num. train time steps", fontdict={"fontsize": 14, "fontweight": "bold"}
#     )
#     ax[2].set_xlabel(
#         "Num. train time steps", fontdict={"fontsize": 14, "fontweight": "bold"}
#     )
#     ax[0].set_ylabel(
#         "Validation MSE Loss", fontdict={"fontsize": 14, "fontweight": "bold"}
#     )
#     ax[0].set_title(f"(A) {model_labels[0]} model", fontdict={"fontsize": 16})
#     ax[1].set_title(f"(B) {model_labels[1]} model", fontdict={"fontsize": 16})
#     ax[2].set_title(f"(C) {model_labels[2]} model", fontdict={"fontsize": 16})

#     # Display the legend in the top right subplot
#     ax[2].legend(
#         handles=legend_code["color_legend"],
#         loc="center right",
#         bbox_to_anchor=(1.5, 0.5),
#         title="Experimental datasets",
#     )

#     # Adjust the layout and display the plot
#     plt.tight_layout()
#     plt.show()


# 5. Cross-dataset Generalization plot
def cross_dataset(experiment_log_folders, model_names, legend_code):
    dataset_labels = legend_code["dataset_labels"]

    analysis_df = pd.DataFrame(
        columns=[
            "experiment_ID",
            "model_type",
            "train_dataset",
            "val_dataset",
            "val_loss",
            "val_baseline",
        ]
    )

    for exp_log_dir, model in zip(experiment_log_folders, model_names):
        for experiment_ID in sorted(
            os.listdir(exp_log_dir), key=lambda x: x.strip("exp_")
        ):
            # Skip if not starts with exp
            if not experiment_ID.startswith("exp") or experiment_ID.startswith("exp_"):
                continue

            val_url = os.path.join(
                exp_log_dir,
                experiment_ID,
                "analysis",
                "validation_loss_per_dataset.csv",
            )
            train_ds_url = os.path.join(
                exp_log_dir, experiment_ID, "dataset", "train_dataset_info.csv"
            )

            val_df = pd.read_csv(val_url)
            val_df["experiment_ID"] = experiment_ID
            val_df["model_type"] = model

            train_dataset_info = pd.read_csv(train_ds_url)
            val_df["train_dataset"] = train_dataset_info["dataset"].unique()[0]

            # Change 'dataset' column name to 'val_dataset'
            val_df = val_df.rename(columns={"dataset": "val_dataset"})

            # Swap uzel and kaplan
            val_df = val_df.iloc[[0, 1, 2, 4, 3, 5, 6], :]
            val_df = val_df.reset_index(drop=True)

            analysis_df = pd.concat([analysis_df, val_df], axis=0)

    train_ds_names = [
        "Leifer2023",
        "Flavell2023",
        "Uzel2022",
        "Yemini2021",
        "Kaplan2020",
        "Skora2018",
        "Nichols2017",
        "Kato2015",
    ]
    val_ds_names = analysis_df["val_dataset"].unique()
    models = analysis_df["model_type"].unique()

    # Figure size
    fig, ax = plt.subplots(1, len(models), figsize=(12, 4), sharex="col", sharey="row")

    # Initialize vmin and vmax for the color scale
    vmin = analysis_df["val_loss"].min()
    vmax = analysis_df["val_loss"].max()

    for i, model_name in enumerate(models):
        # Filter data for the specific model
        df_model_subset = analysis_df[analysis_df["model_type"] == model_name]

        # Create an empty matrix for the heatmap data
        heatmap_data = pd.DataFrame(columns=train_ds_names, index=val_ds_names)

        for train_ds in train_ds_names:
            for val_ds in val_ds_names:
                value = df_model_subset[
                    (df_model_subset["train_dataset"] == train_ds)
                    & (df_model_subset["val_dataset"] == val_ds)
                ]["val_loss"].values
                if value:
                    heatmap_data.at[val_ds, train_ds] = value[0]

        # Plot the heatmap
        sns.heatmap(
            heatmap_data.astype(float),
            cmap="magma_r",
            ax=ax[i],
            annot=True,
            square=True,
            cbar=False,
            vmin=vmin,
            vmax=vmax,
        )
        ax[i].set_title("{}".format(model_name), fontsize=16)
        # Set xlabel
        ax[i].set_xlabel("Train dataset", fontsize=14, fontweight="bold")
        ax[i].set_xticklabels(dataset_labels, rotation=90, fontsize=10)
        # Set ylabel
        ax[0].set_ylabel("Validation dataset", fontsize=14, fontweight="bold")
        ax[i].set_yticklabels(dataset_labels, rotation=0, fontsize=10)

    # Add a single colorbar at the rightmost part
    cbar_ax = fig.add_axes(
        [0.92, 0.125, 0.02, 0.755]
    )  # [left, bottom, width, height] of the colorbar axes in figure coordinates.
    fig.colorbar(
        ax[-1].collections[0], cax=cbar_ax
    )  # ax[-1].collections[0] grabs the colormap of the last subplot

    # Add title to cmap
    cbar_ax.set_ylabel(
        "Validation loss (MSE)",
        fontsize=12,
        fontweight="bold",
        rotation=90,
        labelpad=-57,
    )

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(
        right=0.9
    )  # adjust the rightmost part to make room for the colorbar
    plt.show()


# 6. Prediction gap plots
def prediction_gap(exp_nts_log_dir, legend_code, neuronID, datasetID, wormID):
    ds_color_code = legend_code["ds_color_code"]
    original_ds_color_code = legend_code["original_ds_color_code"]
    model_marker_code = legend_code["model_marker_code"]
    color_legend = legend_code["color_legend"]

    prediction_gap = {
        "experiment_ID": [],
        "dataset": [],
        "gap_mean": [],
        "gap_var": [],
        "num_time_steps": [],
    }

    fig = plt.figure(figsize=(12, 8))  # Create a figure

    # Create 2x2 grid
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.8, 1])

    ax0 = plt.subplot(gs[0, 0])  # First subplot in the first row
    ax1 = plt.subplot(gs[0, 1])  # Second subplot in the first row
    ax2 = plt.subplot(gs[1, 0:2])  # Spanned subplot in the second row

    for exp_dir in np.sort(os.listdir(exp_nts_log_dir)):
        # Skip if not starts with exp
        if not exp_dir.startswith("exp") or exp_dir.startswith("exp_"):
            continue

        for exp_ds in np.sort(
            os.listdir(os.path.join(exp_nts_log_dir, exp_dir, "prediction", "val"))
        ):
            predictions_url = os.path.join(
                exp_nts_log_dir,
                exp_dir,
                "prediction",
                "val",
                exp_ds,
                "worm1",
                "predictions.csv",
            )
            named_neurons_url = os.path.join(
                exp_nts_log_dir,
                exp_dir,
                "prediction",
                "val",
                exp_ds,
                "worm1",
                "named_neurons.csv",
            )

            # Read csv files
            predictions = pd.read_csv(predictions_url)
            named_neurons = pd.read_csv(named_neurons_url)
            neurons = named_neurons["named_neurons"]

            context_data = predictions[predictions["Type"] == "Context"].drop(
                columns=["Type", "Unnamed: 1"]
            )
            context_data = context_data[neurons]  # Filter only named neurons

            ar_gen_data = predictions[predictions["Type"] == "AR Generation"].drop(
                columns=["Type", "Unnamed: 1"]
            )
            ar_gen_data = ar_gen_data[neurons]  # Filter only named neurons

            ground_truth_data = predictions[predictions["Type"] == "Ground Truth"].drop(
                columns=["Type", "Unnamed: 1"]
            )
            ground_truth_data = ground_truth_data[neurons]  # Filter only named neurons

            gt_gen_data = predictions[predictions["Type"] == "GT Generation"].drop(
                columns=["Type", "Unnamed: 1"]
            )
            gt_gen_data = gt_gen_data[neurons]  # Filter only named neurons

            # Compute gap between GT Generation and Ground Truth in the first time step, for all neurons
            gap = np.abs(gt_gen_data.iloc[0, :] - ground_truth_data.iloc[0, :])

            # Retrieve amount of data
            num_time_steps, _, _ = experiment_parameter(
                os.path.join(exp_nts_log_dir, exp_dir), key="time_steps_volume"
            )

            # Save gap statistics
            prediction_gap["experiment_ID"].append(exp_dir)
            prediction_gap["dataset"].append(exp_ds)
            prediction_gap["gap_mean"].append(gap.median())
            prediction_gap["gap_var"].append(gap.var())
            prediction_gap["num_time_steps"].append(num_time_steps)

    prediction_gap = pd.DataFrame(prediction_gap)

    # Plot gap mean with bar errors vs. num_time_steps for all datasets
    # fig, ax = plt.subplots(1,2, figsize=(10, 5))

    dataset_names = prediction_gap["dataset"].unique()

    pred_slopes_info = {
        "dataset": [],
        "slope": [],
        "model_type": [],
    }

    for ds_name in np.sort(dataset_names):
        df_subset = prediction_gap[prediction_gap["dataset"] == ds_name]
        # Plot mean gap with var as yerr, use the color code as in the previous plot
        ax0.errorbar(
            df_subset["num_time_steps"],
            df_subset["gap_mean"],
            yerr=df_subset["gap_var"],
            color=ds_color_code[ds_name[:-4]],
            marker=model_marker_code["LSTM"],
            linestyle="",
        )

        # Try to fit linear regression (log-log)
        try:
            x = np.log(df_subset["num_time_steps"].values)
            y = np.log(df_subset["gap_mean"].values)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            fit_label = (
                "y = {:.2f}x + {:.1f}\n".format(slope, intercept)
                + r"$R^2=$"
                + "{:.3f}".format(r_value**2)
            )
            ax0.plot(
                df_subset["num_time_steps"].values,
                np.exp(intercept + slope * x),
                color=ds_color_code[ds_name[:-4]],
                linestyle="-",
            )
            pred_slopes_info["dataset"].append(ds_name)
            pred_slopes_info["slope"].append(slope)
            pred_slopes_info["model_type"].append("LSTM")

        except:
            pass

    # Set axis labels and title
    ax0.set_xlabel(
        "Time steps per neuron", fontdict={"fontsize": 12, "fontweight": "bold"}
    )
    ax0.set_ylabel("Absolute mean gap", fontdict={"fontsize": 12, "fontweight": "bold"})
    ax0.set_title(
        "(A) LSTM model: evolution of prediction gap\nwith duration of training data",
        fontsize=16,
    )

    # Log-log scale
    ax0.set_xscale("log")
    ax0.set_yscale("log")

    # boxplot slopes
    sns.boxplot(
        data=pred_slopes_info,
        x="model_type",
        y="slope",
        ax=ax1,
        color="lightgrey",
        showfliers=False,
    )
    sns.stripplot(
        data=pred_slopes_info,
        x="model_type",
        y="slope",
        hue="dataset",
        palette=original_ds_color_code,
        size=7,
        ax=ax1,
        dodge=False,
        marker=model_marker_code["LSTM"],
        jitter=False,
    )
    ax1.set_title(
        "(B) LSTM model: variation in prediction\nscaling exponents", fontsize=16
    )
    ax1.set_ylabel(r"Scaling exponent $(e^{slope})$", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Model Architecture", fontsize=12, fontweight="bold")
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()

    # Change legend title
    ax1.get_legend().set_title("Experimental dataset")
    ax1.legend(loc="upper right")
    # Change legend to color_legend
    ax1.get_legend().remove()
    ax1.legend(handles=color_legend, loc="upper right", title="Validation dataset")

    plt.tight_layout(pad=0.5)
    # plt.show()

    worms_to_plot = [wormID]
    neurons_to_plot = [neuronID]
    datasets_to_plot = [datasetID]

    # fig, ax = plt.subplots(figsize=(10, 5))

    for experiment_ID, exp_dir in enumerate(np.sort(os.listdir(exp_nts_log_dir))):
        # Skip if not starts with exp
        if not exp_dir.startswith("exp") or exp_dir.startswith("exp_"):
            continue

        exp_log_dir = os.path.join(exp_nts_log_dir, exp_dir)

        for type_ds in ["val"]:
            for ds_name in os.listdir(os.path.join(exp_log_dir, "prediction", type_ds)):
                for wormID in os.listdir(
                    os.path.join(exp_log_dir, "prediction", type_ds, ds_name)
                ):
                    # Skip if num_worms given
                    if worms_to_plot is not None:
                        if wormID not in worms_to_plot:
                            continue

                    # Skip if dataset given
                    if datasets_to_plot is not None:
                        if ds_name not in datasets_to_plot:
                            continue

                    url = os.path.join(
                        exp_log_dir,
                        "prediction",
                        type_ds,
                        ds_name,
                        wormID,
                        "predictions.csv",
                    )
                    neurons_url = os.path.join(
                        exp_log_dir,
                        "prediction",
                        type_ds,
                        ds_name,
                        wormID,
                        "named_neurons.csv",
                    )

                    # Acess the prediction directory
                    df = pd.read_csv(url)
                    df.set_index(["Type", "Unnamed: 1"], inplace=True)
                    df.index.names = ["Type", ""]

                    # Get the named neurons
                    neurons_df = pd.read_csv(neurons_url)
                    neurons = neurons_df["named_neurons"].tolist()

                    # Treat neurons_to_plot
                    if isinstance(neurons_to_plot, int):
                        neurons = np.random.choice(
                            neurons,
                            size=min(neurons_to_plot, len(neurons)),
                            replace=False,
                        ).tolist()
                    elif isinstance(neurons_to_plot, list):
                        # Skip neurons that are not available
                        neurons = [
                            neuron for neuron in neurons_to_plot if neuron in neurons
                        ]

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
                    gt_generation_color = sns.color_palette(
                        "magma", n_colors=7
                    )  # orange (next time step prediction with gt)
                    ar_generation_color = sns.color_palette(
                        "magma", n_colors=7
                    )  # gree (autoregressive next time step prediction)

                    # logger.info(f'Plotting neuron predictions for {type_ds}/{wormID}...')

                    # Metadata textbox
                    metadata_text = "Signal from {} validation dataset".format(
                        ds_name[:-4]
                    )

                    # Amount of data
                    num_time_steps, _, _ = experiment_parameter(
                        exp_log_dir, key="time_steps_volume"
                    )

                    for neuron in neurons:
                        ax2.plot(
                            time_gt_generated,
                            df.loc["GT Generation", neuron],
                            color=gt_generation_color[experiment_ID],
                        )
                        # ax.plot(time_ar_generated, df.loc['AR Generation', neuron], color=ar_generation_color[experiment_ID], label=num_time_steps)
                        ax2.plot(
                            time_gt_generated,
                            df.loc["GT Generation", neuron],
                            color=gt_generation_color[experiment_ID],
                            label="TSN: {:.2f}".format(num_time_steps),
                        )
                        # ax.plot(time_ar_generated, df.loc['AR Generation', neuron], color=ar_generation_color[experiment_ID], label=num_time_steps)

                    if experiment_ID == 1:
                        up_gap_val = df.loc["Ground Truth", neuron].iloc[0]
                        low_gap_val = df.loc["GT Generation", neuron].iloc[0]

    ax2.plot(
        time_context,
        df.loc["Context", neuron],
        color=gt_color,
        label="Ground truth signal",
    )
    ax2.plot(time_ground_truth, df.loc["Ground Truth", neuron], color=gt_color)

    # Fill the context window
    ax2.axvspan(
        time_context[0],
        time_context[-1],
        alpha=0.1,
        color=gt_color,
        label="Initial context window",
    )

    ax2.set_title(
        f"(C) Teacher forcing generation of {neuron}'s Neuronal Activity", fontsize=16
    )
    ax2.set_xlabel("Time (s)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Activity ($\Delta F / F$)", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right")

    # Add metadata textbox in upper left corner
    ax2.text(
        0.02,
        0.95,
        metadata_text,
        transform=ax2.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round, pad=1", facecolor="white", edgecolor="black", alpha=0.5
        ),
    )

    ax2.set_xlim([0, 240])

    # Vertical line to show gap
    ax2.plot(
        [120, 120],
        [up_gap_val, low_gap_val],
        color="black",
        linestyle="--",
        linewidth=1,
    )
    # Text box with gap
    ax2.text(
        116,
        (up_gap_val + low_gap_val) / 2,
        "Gap".format(up_gap_val - low_gap_val),
        fontsize=10,
        horizontalalignment="center",
        verticalalignment="center",
    )

    plt.tight_layout()

    # Save figure
    plt.show()

    return prediction_gap


# 7a. Prediction plots (autoregresive and teacher forcing side-by-side)
def predictions(experiment_log_folders, model_names, legend_code):
    model_color_code = legend_code["model_color_code"]

    predictions_df = []

    for i, (log_dir, model) in enumerate(zip(experiment_log_folders, model_names)):
        # Access the experiments

        for exp_dir in os.listdir(log_dir):
            # Skip if not starts with exp
            if not exp_dir.startswith("exp") or exp_dir.startswith("exp_"):
                continue

            val_pred_path = os.path.join(log_dir, exp_dir, "prediction", "val")
            train_pred_path = os.path.join(log_dir, exp_dir, "prediction", "train")

            # Loop through all validation datasets
            for pred_path, ds_type in zip(
                [val_pred_path, train_pred_path], ["val", "train"]
            ):
                for ds_name in np.sort(os.listdir(val_pred_path)):
                    # Access validation (or train) predictions
                    pred_url = os.path.join(
                        pred_path, ds_name, "worm1", "predictions.csv"
                    )

                    # Load predictions
                    pred_df = pd.read_csv(pred_url)

                    # Access named neurons
                    # pred_df['named_neurons_filter'] = os.path.join(pred_path, ds_name, 'worm1', 'named_neurons.csv')

                    # Save dataset type
                    pred_df["dataset_type"] = ds_type

                    # Save model
                    pred_df["model_type"] = model

                    # Save experiment parameter
                    num_time_steps, _, _ = experiment_parameter(
                        os.path.join(log_dir, exp_dir), key="num_time_steps"
                    )
                    pred_df["num_time_steps"] = num_time_steps

                    # Save experiment
                    pred_df["experiment_ID"] = exp_dir

                    # Save dataset
                    pred_df["dataset"] = ds_name

                    # Save dataframe
                    predictions_df.append(pred_df)

    predictions_df = pd.concat(predictions_df, axis=0)

    # Plot predictions
    dataset_names = [
        "Kato2015",
        "Nichols2017",
        "Skora2018",
        "Kaplan2020",
        "Uzel2022",
        "Flavell2023",
        "Leifer2023",
    ]
    ds_type = "val"
    models = ["LSTM", "Transformer", "Feedforward"]
    exp = "exp5"  # model trained with maximum amount of data
    neuron_to_plot = "AVER"

    fig, ax = plt.subplots(len(dataset_names), 2, figsize=(20, 5 * len(dataset_names)))

    row_mapping = {
        "Kato2015": 0,
        "Nichols2017": 1,
        "Skora2018": 2,
        "Kaplan2020": 3,
        "Uzel2022": 4,
        "Flavell2023": 5,
        "Leifer2023": 6,
    }

    for subplot_row, ds_name in enumerate(dataset_names):
        metadata_text = "Dataset: {}".format(ds_name[:-4] + " (" + ds_name[-4:] + ")")

        gt_lstm = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "LSTM" and exp == "{exp}" and Type == "Ground Truth"'
        )[neuron_to_plot]
        gt_tr = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "Transformer" and exp == "{exp}" and Type == "Ground Truth"'
        )[neuron_to_plot]
        gt_lin = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "Feedforward" and exp == "{exp}" and Type == "Ground Truth"'
        )[neuron_to_plot]

        gt_gen_lstm = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "LSTM" and exp == "{exp}" and Type == "GT Generation"'
        )[neuron_to_plot]
        gt_gen_tr = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "Transformer" and exp == "{exp}" and Type == "GT Generation"'
        )[neuron_to_plot]
        gt_gen_lin = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "Feedforward" and exp == "{exp}" and Type == "GT Generation"'
        )[neuron_to_plot]

        ar_gen_lstm = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "LSTM" and exp == "{exp}" and Type == "AR Generation"'
        )[neuron_to_plot]
        ar_gen_tr = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "Transformer" and exp == "{exp}" and Type == "AR Generation"'
        )[neuron_to_plot]
        ar_gen_lin = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "Feedforward" and exp == "{exp}" and Type == "AR Generation"'
        )[neuron_to_plot]

        context_lstm = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "LSTM" and exp == "{exp}" and Type == "Context"'
        )[neuron_to_plot]
        context_tr = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "Transformer" and exp == "{exp}" and Type == "Context"'
        )[neuron_to_plot]
        context_lin = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "Feedforward" and exp == "{exp}" and Type == "Context"'
        )[neuron_to_plot]

        max_time_steps = len(context_lstm) + len(gt_lstm) - 1
        time_vector = np.arange(max_time_steps)

        # Teatcher focing plot
        ax[row_mapping[ds_name], 0].plot(
            time_vector[: len(context_lstm)],
            context_lstm,
            color="tab:red",
            label="Ground truth signal",
        )  # Plot context
        ax[row_mapping[ds_name], 0].plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            gt_lstm,
            color="tab:red",
        )  # Plot ground truth
        ax[row_mapping[ds_name], 0].plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            gt_gen_lstm,
            color=model_color_code["LSTM"],
            label="LSTM",
        )  # Plot ground truth generation
        ax[row_mapping[ds_name], 0].plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            gt_gen_tr,
            color=model_color_code["Transformer"],
            label="Transformer",
        )  # Plot ground truth generation
        ax[row_mapping[ds_name], 0].plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            gt_gen_lin,
            color=model_color_code["Feedforward"],
            label="Feedforward",
        )  # Plot ground truth generation

        # Autoregressive plot
        ax[row_mapping[ds_name], 1].plot(
            time_vector[: len(context_lstm)],
            context_lstm,
            color="tab:red",
            label="Ground truth signal",
        )  # Plot context
        ax[row_mapping[ds_name], 1].plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            gt_lstm,
            color="tab:red",
        )  # Plot ground truth
        ax[row_mapping[ds_name], 1].plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            ar_gen_lstm,
            color=model_color_code["LSTM"],
            label="LSTM",
        )  # Plot autoregressive generation
        ax[row_mapping[ds_name], 1].plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            ar_gen_tr,
            color=model_color_code["Transformer"],
            label="Transformer",
        )  # Plot ground truth generation
        ax[row_mapping[ds_name], 1].plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            ar_gen_lin,
            color=model_color_code["Feedforward"],
            label="Feedforward",
        )  # Plot ground truth generation

        # Fill context with tab:blue
        ax[row_mapping[ds_name], 0].axvspan(
            time_vector[0],
            time_vector[len(context_lstm) - 1],
            alpha=0.15,
            color="tab:red",
            label="Initial context window",
        )
        ax[row_mapping[ds_name], 1].axvspan(
            time_vector[0],
            time_vector[len(context_lstm) - 1],
            alpha=0.15,
            color="tab:red",
            label="Initial context window",
        )
        ax[row_mapping[ds_name], 0].set_ylabel(
            "Activity ($\Delta F / F$)", fontsize=14, fontweight="bold"
        )

        ax[row_mapping[ds_name], 0].legend(
            loc="upper left", title=metadata_text, fontsize=12, title_fontsize=12
        )
        ax[row_mapping[ds_name], 1].legend(
            loc="upper left", title=metadata_text, fontsize=12, title_fontsize=12
        )

        # Limit x-axis to 240
        ax[row_mapping[ds_name], 0].set_xlim([0, 240])
        ax[row_mapping[ds_name], 1].set_xlim([0, 240])

    ax[-1, 0].set_xlabel("Time steps (s)", fontsize=14, fontweight="bold")
    ax[-1, 1].set_xlabel("Time steps (s)", fontsize=14, fontweight="bold")

    ax[0, 0].set_title(
        f"Teacher forcing generation of {neuron_to_plot}'s Neuronal Activity",
        fontsize=16,
    )
    ax[0, 1].set_title(
        f"Autoregressive generation of {neuron_to_plot}'s Neuronal Activity",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()

    return predictions_df


# 7b. (Only) Teacher forcing prediction plots
def teacher_forcing(
    experiment_log_folders, model_names, legend_code, ds_type, exp, neuron_to_plot
):
    model_color_code = legend_code["model_color_code"]

    predictions_df = []

    for i, (log_dir, model) in enumerate(zip(experiment_log_folders, model_names)):
        # Access the experiments

        for exp_dir in os.listdir(log_dir):
            # Skip if not starts with exp
            if not exp_dir.startswith("exp") or exp_dir.startswith("exp_"):
                continue

            val_pred_path = os.path.join(log_dir, exp_dir, "prediction", "val")
            train_pred_path = os.path.join(log_dir, exp_dir, "prediction", "train")

            # Loop through all validation datasets
            for pred_path, ds_type in zip(
                [val_pred_path, train_pred_path], ["val", "train"]
            ):
                for ds_name in np.sort(os.listdir(val_pred_path)):
                    # Access validation (or train) predictions
                    pred_url = os.path.join(
                        pred_path, ds_name, "worm1", "predictions.csv"
                    )

                    # Load predictions
                    pred_df = pd.read_csv(pred_url)

                    # Access named neurons
                    # pred_df['named_neurons_filter'] = os.path.join(pred_path, ds_name, 'worm1', 'named_neurons.csv')

                    # Save dataset type
                    pred_df["dataset_type"] = ds_type

                    # Save model
                    pred_df["model_type"] = model

                    # Save experiment parameter
                    num_time_steps, _, _ = experiment_parameter(
                        os.path.join(log_dir, exp_dir), key="num_time_steps"
                    )
                    pred_df["num_time_steps"] = num_time_steps

                    # Save experiment
                    pred_df["experiment_ID"] = exp_dir

                    # Save dataset
                    pred_df["dataset"] = ds_name

                    # Save dataframe
                    predictions_df.append(pred_df)

    predictions_df = pd.concat(predictions_df, axis=0)

    dataset_names = [
        "Kato2015",
        "Skora2018",
        "Nichols2017",
        "Kaplan2020",
        "Uzel2022",
        "Flavell2023",
        "Leifer2023",
    ]

    fig = plt.figure(
        figsize=(20, 5 * len(dataset_names) // 3)
    )  # Adjusted to account for 3 columns.

    row_mapping = {
        "Kato2015": (0, 0),
        "Skora2018": (0, 1),
        "Nichols2017": (0, 2),
        "Kaplan2020": (0, 2),
        "Uzel2022": (1, 0),
        "Flavell2023": (1, 1),
        "Leifer2023": (1, 2),
    }

    for subplot_row, ds_name in enumerate(dataset_names):
        metadata_text = "Dataset: {}".format(ds_name[:-4] + " (" + ds_name[-4:] + ")")

        gt_lstm = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "LSTM" and exp == "{exp}" and Type == "Ground Truth"'
        )[neuron_to_plot]

        gt_gen_lstm = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "LSTM" and exp == "{exp}" and Type == "GT Generation"'
        )[neuron_to_plot]
        gt_gen_tr = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "Transformer" and exp == "{exp}" and Type == "GT Generation"'
        )[neuron_to_plot]
        gt_gen_lin = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "Feedforward" and exp == "{exp}" and Type == "GT Generation"'
        )[neuron_to_plot]

        context_lstm = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "LSTM" and exp == "{exp}" and Type == "Context"'
        )[neuron_to_plot]

        max_time_steps = len(context_lstm) + len(gt_lstm) - 1
        time_vector = np.arange(max_time_steps)

        # Get the corresponding row and col index for the subplot.
        row, col = row_mapping[ds_name]
        ax = plt.subplot2grid(
            (len(dataset_names) // 3, 3), (row, col)
        )  # 3 columns subplot grid

        # Teacher forcing plot
        ax.plot(
            time_vector[: len(context_lstm)],
            context_lstm,
            color="tab:red",
            label="Ground truth signal",
            linewidth=2,
        )
        ax.plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            gt_lstm,
            color="tab:red",
            linewidth=2,
        )
        ax.plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            gt_gen_lstm,
            color=model_color_code["LSTM"],
            label="LSTM",
        )
        ax.plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            gt_gen_tr,
            color=model_color_code["Transformer"],
            label="Transformer",
        )
        ax.plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            gt_gen_lin,
            color=model_color_code["Feedforward"],
            label="Feedforward",
        )

        # Baseline model prediction
        baseline_activity = np.zeros(len(gt_lstm))
        baseline_activity[0] = context_lstm.iloc[-2]
        baseline_activity[1:] = gt_lstm[:-1]
        ax.plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            baseline_activity,
            color="black",
            linestyle="--",
            label="Baseline model",
        )

        ax.axvspan(
            time_vector[0],
            time_vector[len(context_lstm) - 1],
            alpha=0.15,
            color="tab:red",
            label="Initial context window",
        )
        ax.set_ylabel("Activity ($\Delta F / F$)", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", title=metadata_text, fontsize=12, title_fontsize=12)
        ax.set_xlim([0, 240])

        # Only set ylabel for the leftmost plots
        if col == 0:
            ax.set_ylabel("Activity ($\Delta F / F$)", fontsize=14, fontweight="bold")
        else:
            ax.set_ylabel("")

        # Only set xlabel for the bottom center plot
        if row == (len(dataset_names) // 3 - 1) and col == 1:
            ax.set_xlabel("Time steps (s)", fontsize=14, fontweight="bold")

        # Only place the legend in the right-top plot
        if row == 0 and col == 2:
            ax.legend(
                loc="upper right", title=metadata_text, fontsize=12, title_fontsize=12
            )
        else:
            ax.legend().remove()
            # Add a text box with the dataset metadata
            ax.text(
                0.95,
                0.95,
                metadata_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round, pad=0.5",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.5,
                ),
            )

        # Set title for the middle plot of the first row, if needed.
        if row == 0 and col == 1:
            ax.set_title(
                f"Teacher forcing generation of {neuron_to_plot}'s Neuronal Activity",
                fontsize=16,
            )

    # plt.tight_layout()
    plt.show()


# 7c. (Only) Autoregressive prediction plots
def autoregressive(
    experiment_log_folders, model_names, legend_code, ds_type, exp, neuron_to_plot
):
    model_color_code = legend_code["model_color_code"]

    predictions_df = []

    for i, (log_dir, model) in enumerate(zip(experiment_log_folders, model_names)):
        # Access the experiments

        for exp_dir in os.listdir(log_dir):
            # Skip if not starts with exp
            if not exp_dir.startswith("exp") or exp_dir.startswith("exp_"):
                continue

            val_pred_path = os.path.join(log_dir, exp_dir, "prediction", "val")
            train_pred_path = os.path.join(log_dir, exp_dir, "prediction", "train")

            # Loop through all validation datasets
            for pred_path, ds_type in zip(
                [val_pred_path, train_pred_path], ["val", "train"]
            ):
                for ds_name in np.sort(os.listdir(val_pred_path)):
                    # Access validation (or train) predictions
                    pred_url = os.path.join(
                        pred_path, ds_name, "worm1", "predictions.csv"
                    )

                    # Load predictions
                    pred_df = pd.read_csv(pred_url)

                    # Access named neurons
                    # pred_df['named_neurons_filter'] = os.path.join(pred_path, ds_name, 'worm1', 'named_neurons.csv')

                    # Save dataset type
                    pred_df["dataset_type"] = ds_type

                    # Save model
                    pred_df["model_type"] = model

                    # Save experiment parameter
                    num_time_steps, _, _ = experiment_parameter(
                        os.path.join(log_dir, exp_dir), key="num_time_steps"
                    )
                    pred_df["num_time_steps"] = num_time_steps

                    # Save experiment
                    pred_df["experiment_ID"] = exp_dir

                    # Save dataset
                    pred_df["dataset"] = ds_name

                    # Save dataframe
                    predictions_df.append(pred_df)

    predictions_df = pd.concat(predictions_df, axis=0)

    dataset_names = [
        "Kato2015",
        "Skora2018",
        "Nichols2017",
        "Kaplan2020",
        "Uzel2022",
        "Flavell2023",
        "Leifer2023",
    ]

    fig = plt.figure(
        figsize=(20, 5 * len(dataset_names) // 3)
    )  # Adjusted to account for 3 columns.

    row_mapping = {
        "Kato2015": (0, 0),
        "Skora2018": (0, 1),
        "Nichols2017": (0, 2),
        "Kaplan2020": (0, 2),
        "Uzel2022": (1, 0),
        "Flavell2023": (1, 1),
        "Leifer2023": (1, 2),
    }

    for subplot_row, ds_name in enumerate(dataset_names):
        metadata_text = "Dataset: {}".format(ds_name[:-4] + " (" + ds_name[-4:] + ")")

        gt_lstm = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "LSTM" and exp == "{exp}" and Type == "Ground Truth"'
        )[neuron_to_plot]

        ar_gen_lstm = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "LSTM" and exp == "{exp}" and Type == "AR Generation"'
        )[neuron_to_plot]
        ar_gen_tr = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "Transformer" and exp == "{exp}" and Type == "AR Generation"'
        )[neuron_to_plot]
        ar_gen_lin = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "Feedforward" and exp == "{exp}" and Type == "AR Generation"'
        )[neuron_to_plot]

        context_lstm = predictions_df.query(
            f'dataset_type == "{ds_type}" and dataset == "{ds_name}" and model == "LSTM" and exp == "{exp}" and Type == "Context"'
        )[neuron_to_plot]

        max_time_steps = len(context_lstm) + len(gt_lstm) - 1
        time_vector = np.arange(max_time_steps)

        # Get the corresponding row and col index for the subplot.
        row, col = row_mapping[ds_name]
        ax = plt.subplot2grid(
            (len(dataset_names) // 3, 3), (row, col)
        )  # 3 columns subplot grid

        # Autoregressive plot
        ax.plot(
            time_vector[: len(context_lstm)],
            context_lstm,
            color="tab:red",
            label="Ground truth signal",
            linewidth=2,
        )
        ax.plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            gt_lstm,
            color="tab:red",
            linewidth=2,
        )
        ax.plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            ar_gen_lstm,
            color=model_color_code["LSTM"],
            label="LSTM",
        )
        ax.plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            ar_gen_tr,
            color=model_color_code["Transformer"],
            label="Transformer",
        )
        ax.plot(
            time_vector[len(context_lstm) - 1 : max_time_steps],
            ar_gen_lin,
            color=model_color_code["Feedforward"],
            label="Feedforward",
        )

        ax.axvspan(
            time_vector[0],
            time_vector[len(context_lstm) - 1],
            alpha=0.15,
            color="tab:red",
            label="Initial context window",
        )
        ax.set_ylabel("Activity ($\Delta F / F$)", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", title=metadata_text, fontsize=12, title_fontsize=12)
        ax.set_xlim([0, 240])

        # Only set ylabel for the leftmost plots
        if col == 0:
            ax.set_ylabel("Activity ($\Delta F / F$)", fontsize=14, fontweight="bold")
        else:
            ax.set_ylabel("")

        # Only set xlabel for the bottom center plot
        if row == (len(dataset_names) // 3 - 1) and col == 1:
            ax.set_xlabel("Time steps (s)", fontsize=14, fontweight="bold")

        # Only place the legend in the right-top plot
        if row == 0 and col == 2:
            ax.legend(
                loc="upper right", title=metadata_text, fontsize=12, title_fontsize=12
            )
        else:
            ax.legend().remove()
            # Add a text box with the dataset metadata
            ax.text(
                0.95,
                0.95,
                metadata_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round, pad=0.5",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.5,
                ),
            )

        # Set title for the middle plot of the first row, if needed.
        if row == 0 and col == 1:
            ax.set_title(
                f"Autoregressive generation of {neuron_to_plot}'s Neuronal Activity",
                fontsize=16,
            )

    # plt.tight_layout()
    plt.show()
