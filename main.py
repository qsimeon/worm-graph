from pkg import *


@hydra.main(version_base=None, config_path="conf", config_name="main")
def pipeline(
    config: DictConfig,
) -> None:
    """Runs a complete pipeline to train a model and make predictions.

    Can be configured using the main.yaml file in the conf directory.

    Parameters
    ----------
    config: DictConfig
        Hydra configuration object.

    Calls
    -----
    process_data : function in preprocess/_main.py
        Configuration file in conf/preprocess.yaml

    get_dataset : function in datasets/_main.py
        Configuration file in conf/dataset.yaml

    get_model : function in models/_main.py
        Configuration file in conf/model.yaml

    train_model : function in train/_main.py
        Configuration file in conf/train.yaml

    plot_figures : function in visualization/_main.py
        Configuration file in conf/visualize.yaml

    TODO: Implement `analyze_outputs` : function in analysis/_main.py; config in conf/analysis.yaml

    Notes
    -----
    * Use mode: RUN if you are having a UserWarning with MULTIRUN

    """
    # Display Pytorch device
    torch_device()

    # Intialize random seeds
    init_random_seeds(config.globals.random_seed)

    # Skips if data already preprocessed
    #! process_data(config)

    # Returns a generator of single worm datasets
    dataset = get_dataset(config)

    # Get the model to train
    model = get_model(config)

    # Train model is the bulk of the pipeline code
    model, log_dir, config = train_model(
        config,
        model,
        dataset,
        shuffle_worms=config.globals.shuffle_worms,  # shuffle worms
        log_dir=None,  # hydra changes working directory to log directory
    )

    # Use trained model to make predictions on the dataset
    make_predictions(
        config,  # `train_model` injected the `predict` params into config`
        model=None,
        dataset=None,
        log_dir=log_dir,
        use_residual=config.globals.use_residual,
        smooth_data=config.globals.smooth_data,
    )

    # Plot figures
    plot_figures(config, log_dir)

    ## TODO: Analysis of outputs
    # analyze_outputs(config, log_dir)

    # Free up GPU
    clear_cache()

    return None


if __name__ == "__main__":
    pipeline()
