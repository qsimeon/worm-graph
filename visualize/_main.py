from visualize._utils import *


def plot_figures(
    visualize_config: DictConfig,
) -> None:
    """
    Plots the loss curves and other basic plots from the results of traininig
    a model on a worm neural activity dataset. Save figures to the directory `log_dir`.
    """

    log_dir = visualize_config.log_dir # If none, use current working directory
    if log_dir is None:
        log_dir = os.getcwd()  # logs/hydra/${now:%Y_%m_%d_%H_%M_%S}

    worm = visualize_config.worm
    neuron = visualize_config.neuron
    use_residual = visualize_config.use_residual

    # loss curves
    plot_loss_curves(log_dir)

    # scatterplot of modeled vs. real neuron activity
    plot_correlation_scatterplot(
        log_dir,
        worm,
        neuron,
        use_residual,
    )

    # calcium residuals
    plot_targets_predictions(
        log_dir,
        worm,
        neuron,
        use_residual,
    )

    # plot model weights
    plot_before_after_weights(log_dir)

    # Delete the new log directory created by hydra if using a specific log_dir
    if visualize_config.log_dir is not None:
        cur_dir = os.getcwd()
        os.chdir("..")
        os.rmdir(cur_dir)

    # TODO add more plotting functions for different types of figures

    return None


if __name__ == "__main__":
    config = OmegaConf.load("configs/submodule/visualize.yaml")
    print(OmegaConf.to_yaml(config), end="\n\n")

    # Create new to log directory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join("logs/hydra", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # Move to new log directory
    os.chdir(log_dir)

    plot_figures(config.visualize)
