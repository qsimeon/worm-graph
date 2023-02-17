from visualization._utils import *


def plot_figures(
    config: DictConfig,
    log_dir: str = None,
    worm: str = None,
    neuron: str = None,
) -> None:
    """
    Plots the loss curves and other basic plots from the results of traininig
    a model on a worm neural activity dataset. Saves figures to the directory `log_dir`.
    """
    # get default hyperparams for plotting
    if log_dir is None:
        log_dir = config.visualize.default_log_dir
    if worm is None:
        worm = config.visualize.default_worm
    if neuron is None:
        neuron = config.visualize.default_neuron

    # loss curves
    plot_loss_curves(log_dir)

    # calcium residuals
    plot_targets_predictions(log_dir, worm, neuron)

    # scatterplot of modelled v. real neuron activity
    plot_correlation_scatterplot(log_dir, worm, neuron)
    # TODO add more figures

    return None


if __name__ == "__main__":
    config = OmegaConf.load("conf/visualize.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    plot_figures(config)
