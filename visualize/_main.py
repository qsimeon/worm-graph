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

    log_dir = visualize_config.plot_figures_from_this_log_dir

    assert (
        log_dir is not None
    ), "log_dir is None. Please specify a log directory to plot figures from."

    # Convert relative path to absolute path if necessary
    log_dir = os.path.join(ROOT_DIR, log_dir) if not os.path.isabs(log_dir) else log_dir

    # Check if the log directory exists
    assert os.path.exists(log_dir), f"log directory {log_dir} does not exist."

    # Loop through submodules log directories
    for submodule_dir in os.listdir(log_dir):
        if submodule_dir == "dataset":
            logger.info("Plotting submodule 'dataset'.")
            plot_dataset_info(log_dir=log_dir)

        if submodule_dir == "train":
            logger.info("Plotting submodule 'train'.")
            plot_loss_curves(log_dir=log_dir, info_to_display=None)

        if submodule_dir == "prediction":
            logger.info("Plotting submodule 'prediction'.")
            plot_predictions(
                log_dir=log_dir,
                neurons_to_plot=visualize_config.predict.neurons_to_plot,
                worms_to_plot=visualize_config.predict.worms_to_plot,
            )
            # plot_pca_trajectory(
            #     log_dir=log_dir,
            #     worms_to_plot=visualize_config.predict.worms_to_plot,
            #     plot_type="3D",
            # ) # DEBUG
            plot_pca_trajectory(
                log_dir=log_dir,
                worms_to_plot=visualize_config.predict.worms_to_plot,
                plot_type="2D",
            )

        if submodule_dir == "analysis":
            logger.info("Plotting submodule 'analysis'.")
            plot_loss_per_dataset(log_dir=log_dir, mode="validation")


def plot_experiment(visualize_config: DictConfig, exp_config: DictConfig) -> None:
    """
    Plots the scaling laws for the worm neural activity dataset.
    """

    try:
        log_dir = visualize_config.plot_figures_from_this_log_dir

        assert (
            log_dir is not None
        ), "log_dir is None. Please specify a log directory to plot figures from."

        # If this log is an experiment log, it contains 'exp0' directory
        if "exp0" in os.listdir(log_dir):
            exp_log_dir = log_dir
        else:
            # One directory up if inside any 'expN' directory
            log_dir = os.path.dirname(log_dir)
            if "exp0" in os.listdir(log_dir):
                exp_log_dir = log_dir
            else:
                logger.info(
                    f"Log directory {log_dir} is not an experiment log. Skipping experiment plots."
                )
                return None

        # Create directory to store experiment plots
        exp_plot_dir = os.path.join(exp_log_dir, "exp_plots")
        os.makedirs(exp_plot_dir, exist_ok=True)

        # Get experiment key
        pipeline_info_exp0 = OmegaConf.load(
            os.path.join(exp_log_dir, "exp0", "pipeline_info.yaml")
        )
        exp_key = pipeline_info_exp0.experiment.key

        value, title, xaxis = experiment_parameter(
            os.path.join(exp_log_dir, "exp0"), key=exp_key
        )
        if value is None:
            logger.info(
                f"Experiment {exp_key} not found in {exp_log_dir}. Skipping experiment plots."
            )
            return None

        logger.info(f"Plotting experiment {exp_key}.")

        # Plot loss curves
        plot_experiment_losses(
            exp_log_dir=exp_log_dir,
            exp_key=exp_key,
            exp_plot_dir=exp_plot_dir,
        )

        # Plot summary statistics
        plot_experiment_summaries(
            exp_log_dir=exp_log_dir,
            exp_key=exp_key,
            exp_plot_dir=exp_plot_dir,
        )

        # Plot validation loss per individual dataset
        plot_experiment_loss_per_dataset(
            exp_log_dir=exp_log_dir,
            exp_key=exp_key,
            exp_plot_dir=exp_plot_dir,
            mode="validation",
        )

    except Exception as e:
        logger.info(f"Not all experiments are finished. Skipping for now.")
        logger.error(f"The error that occurred: {e}")
        logger.error(traceback.format_exc())  # This will print the full traceback
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
