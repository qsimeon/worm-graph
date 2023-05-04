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

    TODO: analyze_outputs : function in analysis/_main.py
    TODO:    Configuration file in conf/analysis.yaml

    Notes
    -----
    * Use mode: RUN if you are having a UserWarning with MULTIRUN

    """
    # Print Pytorch device
    print("\ntorch device: %s" % (DEVICE), end="\n\n")

    # Intialize random seeds
    init_random_seeds(config.globals.random_seed)

    # Skips if data already preprocessed
    process_data(config)

    # Returns a generator of single worm datasets
    dataset = get_dataset(config)

    # Get the model to train
    model = get_model(config)

    # train model is the bulk of the pipeline code
    model, log_dir = train_model(
        config,
        model,
        dataset,
        shuffle=config.globals.shuffle,  # shuffle worms
        log_dir=None,  # hydra changes working directory to log directory
    )

    # use trained model to make predictions on the dataset
    make_predictions(
        config,
        model,
        dataset,
        log_dir,
        use_residual=config.globals.use_residual,
        smooth_data=config.globals.smooth_data,
    )

    # plot figures
    plot_figures(config, log_dir)

    ## TODO: analysis
    # analyze_outputs(config, log_dir)
    return None


if __name__ == "__main__":
    pipeline()