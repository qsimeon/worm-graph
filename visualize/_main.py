from visualize._utils import *

# Init logger
logger = logging.getLogger(__name__)

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

    logger.info('Start plotting figures.')

    # Plots related to training (loss_curves.csv file exists, checkpoints exists)
    if 'train' in pipeline_info.submodule:
        # Loss curves
        plot_loss_curves(log_dir)

        # Model weights
        #! plot_before_after_weights(log_dir) # TODO: modify to work with EarlyStopping

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

def plot_experiment(log_dir, exp_config: DictConfig) -> None:
    """
    Plots the scaling laws for the worm neural activity dataset.
    """

    # Scaling law plots
    scaling_law_dir = os.path.join(log_dir, "scaling_laws")
    os.makedirs(scaling_law_dir, exist_ok=True)

    # Computation time vs. key (amount of data, hidden size, etc.)
    df, fig, ax = seconds_per_epoch_plot(exp_log_dir = log_dir,
                                         key = exp_config.name,
                                         log_scale = exp_config.options.log_scale)
    
    # All test losses
    df, fig, ax = test_losses_plot(exp_log_dir = log_dir,
                                   key = exp_config.name,
                                   threshold = 1e-5,
                                   window = 30,
                                   xlim = None)
    
    # Test loss vs. key (amount of data, hidden size, etc.)
    df, fig, ax = scaling_law_plot(exp_log_dir = log_dir,
                                   key = exp_config.name,
                                   log_scale = exp_config.options.log_scale)

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
