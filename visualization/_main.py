from visualization._utils import *


def plot_figures(
    config: DictConfig,
    log_dir: str = None,
    worm: str = None,
    neuron: str = None,
) -> None:
    """
    Plots the loss curves and other basic plots from the results of traininig
    a model on a worm neural activity dataset. Save figures to the directory `log_dir`.
    """
    # get default hyperparams for plotting
    if log_dir is None:
        log_dir = config.visualize.log_dir
    if worm is None:
        worm = config.visualize.worm
    if neuron is None:
        neuron = config.visualize.neuron
    # get other (global) config params
    if config.get("globals"):
        use_residual = config.globals.use_residual
    else:
        use_residual = False

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

    # TODO add more plotting functions for different figures

    return None


if __name__ == "__main__":
    config = OmegaConf.load("conf/visualize.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    plot_figures(config)
