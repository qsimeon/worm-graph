from analysis._pkg import *


def find_config_files(root_dir):
    for file in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file)
        if file == "config.yaml":
            yield file_path
        elif os.path.isdir(file_path) and not file.startswith("."):
            for config_file in find_config_files(file_path):
                yield config_file


# TODO: rewrite function below for arbitrary nested configs
def get_config_value(config, key):
    if "." in key:
        key_parts = key.split(".")
        subconfig = config
        for part in key_parts:
            if part in subconfig:
                subconfig = subconfig[part]
            else:
                return None
        return subconfig
    else:
        return config.get(key)


def plot_loss_vs_parameter(
    config_dir, varied_param, control_param, subplot_param, ax=None
):
    """
    Plots the minimum validation loss against a varied parameter for different levels of a control parameter.
    Creates a separate subplot for each unique value of a third subplot parameter.

    Args:
        config_dir (str): Directory containing the config files.
        varied_param (str): Parameter that is varied across the experiments.
        control_param (str): Parameter that controls the color of the plot lines.
        subplot_param (str): Parameter that determines the creation of separate subplots.
        ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto, defaults to None.

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

    # Create a new DataFrame to hold the smoothed data
    smoothed_df = pd.DataFrame()

    # Loop over each unique value of the control parameter
    for lvl in df[".".join(control_param)].unique():
        # And each unique value of the subplot parameter
        for sub in df[".".join(subplot_param)].unique():
            # Filter the DataFrame for this subset of the data
            subset_df = df[
                (df[".".join(control_param)] == lvl)
                & (df[".".join(subplot_param)] == sub)
            ]

            # Apply the LOWESS function to the data
            smooth = lowess(subset_df["loss"], subset_df[".".join(varied_param)])

            # Add the smoothed data to the new DataFrame
            smoothed_subset_df = pd.DataFrame(
                smooth, columns=[".".join(varied_param), "loss"]
            )
            smoothed_subset_df[".".join(control_param)] = lvl
            smoothed_subset_df[".".join(subplot_param)] = sub
            smoothed_df = pd.concat([smoothed_df, smoothed_subset_df])

    df = smoothed_df

    # Create a grid of subplots, with one row for each unique value of 'subplot_param'
    # The grid will share the y-axis across all subplots
    g = sns.FacetGrid(df, row=".".join(subplot_param), height=3, aspect=4, sharey=True)

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
    g.fig.subplots_adjust(top=0.9)  # Adjust the Figure in `g`
    g.fig.suptitle(
        "Scaling plot: loss vs {} {} \n Validation loss after training on different {} {}".format(
            *varied_param, *control_param
        )
    )

    # Add a legend to the figure
    g.add_legend()

    # Set the y-axis limits for all subplots
    g.set(ylim=(-0.1, None))

    # Set x-axis scale to log if the varied parameter is train_size or hidden_size
    if varied_param[1] in {"worm_timesteps", "hidden_size"}:
        g.set(xscale="log")

    # Save the figure as an image if 'ax' is not provided
    if ax is None:
        if not os.path.exists("figures"):
            os.mkdir("figures")
        g.savefig("figures/scaling_plot_val_loss_vs_{}_{}.png".format(*varied_param))

    # Return the DataFrame
    return df
