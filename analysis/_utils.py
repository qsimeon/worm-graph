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
    config_dir,
    vary_param,
    contrast_param,
    repeats_param,
    ax=None,
):
    """
    Plots the minimum validation loss against a varied parameter (vary_param)
    at different levels of another control parameter (contrast_param) using Seaborn.
    The repeats_param parameter is used for the individual representations.

    Args:
        config_dir: The directory containing the config files.
        vary_param: The independent variable.
        contrast_param: The control variable.
        repeats_param: The parameter to use for repetitions.

    Returns:
        df: A Pandas data frame containing the data used to create the plot.
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
    vary_param = vary_param.split(".")
    contrast_param = contrast_param.split(".")
    repeats_param = repeats_param.split(".")

    # Create a data frame with the relevant data
    records = []
    for cfg_path, loss_cfg_tuple in configs.items():
        lvl = loss_cfg_tuple[1][contrast_param[0]][contrast_param[1]]
        val = loss_cfg_tuple[1][vary_param[0]][vary_param[1]]
        rep = loss_cfg_tuple[1][repeats_param[0]][repeats_param[1]]
        loss = loss_cfg_tuple[0]
        records.append((lvl, val, rep, loss))

    df = pd.DataFrame(
        records,
        columns=[
            ".".join(contrast_param),
            ".".join(vary_param),
            ".".join(repeats_param),
            "loss",
        ],
    )

    # TODO: Why was this here?
    # df[".".join(contrast_param)] = "None"

    # Create the Seaborn plot
    ax_ = ax
    sns.set(style="white", font_scale=1.2)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x=".".join(vary_param),
        y="loss",
        hue=".".join(contrast_param),
        units=".".join(repeats_param),
        estimator=None,
        errorbar="se",
        err_style="band",
        marker="o",
        ax=ax,
        # TODO: Why was this False before?
        legend=True, 
    )

    # Customize plot labels and title
    ax.set_xlabel("%s %s" % (vary_param[0], vary_param[1]))
    ax.set_ylabel("Validation Loss")
    ax.set_title(
        "Scaling plot: loss vs {} {} \n Validation loss after training on different {} {}".format(
            *vary_param, *contrast_param
        )
    )

    # Set the y-axis limits
    plt.ylim([-0.1, None])

    # Set x-axis scale to log if the varied parameter is train_size or hidden_size
    if vary_param[1] in {"worm_timesteps", "hidden_size"}:
        ax.set_xscale("log")

    # Save the plot as an image
    if ax_ is None:
        if not os.path.exists("figures"):
            os.mkdir("figures")
        plt.savefig("figures/scaling_plot_val_loss_vs_{}_{}.png".format(*vary_param))

    # Return the data frame
    return df


# TODO: write a function that categorizes neurons into sensory, inter, and motor