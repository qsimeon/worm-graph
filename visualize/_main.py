from visualize._utils import *


def plot_figures(
    visualize_config: DictConfig,
) -> None:
    """
    Plots the loss curves and other basic plots from the results of traininig
    a model on a worm neural activity dataset. Save figures to the directory `log_dir`.
    """

    log_dir = visualize_config.log_dir

    worm = visualize_config.worm
    neuron = visualize_config.neuron
    use_residual = visualize_config.use_residual

    # Load pipeline info
    pipeline_info = OmegaConf.load(os.path.join(log_dir, "pipeline_info.yaml"))

    # Plots related to training (loss_curves.csv file exists, checkpoints exists)
    if 'train' in pipeline_info.submodule:
        # Loss curves
        plot_loss_curves(log_dir)

        # Model weights
        plot_before_after_weights(log_dir)

    # Plots reletad to predictions (at least worm0 directory exists)
    if 'predict' in pipeline_info.submodule:
        # Correlation plot
        plot_correlation_scatterplot(
            log_dir,
            worm,
            neuron,
            use_residual,
        )

        # Generation plot
        plot_targets_predictions(
            log_dir,
            worm,
            neuron,
            use_residual,
        )

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
