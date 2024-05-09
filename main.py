from pkg import *


@hydra.main(version_base=None, config_path="configs", config_name="pipeline")
def pipeline(cfg: DictConfig) -> None:
    """Main custom pipeline."""
    # Set log directory
    log_dir = os.getcwd()
    # Init logger
    logger = logging.getLogger(__name__)
    # Only the pipeline module
    if len(cfg) == 1:
        raise ValueError(
            "No submodules in the pipeline. Run python main.py +experiment=your_experiment"
        )
    # Need to have a model and a dataset
    if "train" in cfg.submodule:
        if "model" not in cfg.submodule:
            raise AssertionError("Need to specify a model before training.")
        if "dataset" not in cfg.submodule:
            raise AssertionError("Need to specify a dataset before training.")
    # Performing predictions on current or existing logs
    if "predict" in cfg.submodule:
        # Predicting on a existing log folder
        if cfg.submodule.predict.predict_this_log_dir is not None:
            log_dir = cfg.submodule.predict.predict_this_log_dir
            OmegaConf.update(
                cfg.submodule,
                "model.use_this_pretrained_model",
                os.path.join(log_dir, "train", "checkpoints", "model_best.pt"),
                force_add=True,
            )
        else:
            # Need to have a model and a dataset
            if "model" not in cfg.submodule:
                raise AssertionError("Need to specify a model before predicting.")
            if "dataset" not in cfg.submodule:
                raise AssertionError("Need to specify a dataset before predicting.")
    # Some setup (get device, clear cache, set random seeds, etc.)
    logger.info("Torch device: %s" % (DEVICE))
    torch.cuda.empty_cache()
    if cfg.experiment.seed is None:
        cfg.experiment.seed = random.randint(0, 100)
    logger.info("Setting random seeds to %d" % (cfg.experiment.seed))
    init_random_seeds(cfg.experiment.seed)  # Set random seeds
    # Run each submodule in sequence
    # PREPROCESS
    if "preprocess" in cfg.submodule:
        # Preprocess the raw neural and connectome data
        process_data(cfg.submodule.preprocess)
    # DATASET
    if "dataset" in cfg.submodule:
        # Create the train and validation datasets
        train_dataset, val_dataset = get_datasets(
            cfg.submodule.dataset, save=cfg.submodule.dataset.save_datasets
        )
        # Update visualize submodule to plot current run
        if "visualize" in cfg.submodule:
            cfg.submodule.visualize.plot_this_log_dir = log_dir
    # MODEL
    if "model" in cfg.submodule:
        # Initialize a new model or get pretrained model
        model = get_model(cfg.submodule.model)
    # TRAIN
    if "train" in cfg.submodule:
        # Run the training loop
        model, metric = train_model(
            train_config=cfg.submodule.train,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            verbose=False,
        )
        # Update visualize and analysis submodule to plot current run
        if "analysis" in cfg.submodule:
            cfg.submodule.analysis.analyse_this_log_dir = log_dir
        if "visualize" in cfg.submodule:
            cfg.submodule.visualize.plot_this_log_dir = log_dir
    # PREDICT
    if "predict" in cfg.submodule:
        # Use trained model to make predictions
        make_predictions(
            predict_config=cfg.submodule.predict,
            model=model,
        )
        # Update visualize and analysis submodule to plot current run
        if "analysis" in cfg.submodule:
            cfg.submodule.analysis.analyse_this_log_dir = log_dir
        if "visualize" in cfg.submodule:
            cfg.submodule.visualize.plot_this_log_dir = log_dir
    # ANALYSIS
    if "analysis" in cfg.submodule:
        # Analyze the results of the experiment
        analyse_run(
            analysis_config=cfg.submodule.analysis,
        )
    # VISUALIZE
    if "visualize" in cfg.submodule:
        # Plot figures for the experiment
        plot_figures(
            visualize_config=cfg.submodule.visualize,
        )
        plot_experiment(visualize_config=cfg.submodule.visualize, exp_config=cfg.experiment)
    # Clear GPU cache and ave pipeline config which may have been modified
    torch.cuda.empty_cache()
    OmegaConf.save(cfg, os.path.join(log_dir, "pipeline_info.yaml"))
    # Return metric for Optuna automatic hyperparameter tuning
    if "train" in cfg.submodule:
        logger.info("Experiment finished. Final metric: %s" % (metric))
        return metric
    return None


if __name__ == "__main__":
    pipeline()
