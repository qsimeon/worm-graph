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

    # TODO add more plotting functions for different types of figures

    return None


if __name__ == "__main__":
    config = OmegaConf.load("conf/visualize.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    plot_figures(config)
