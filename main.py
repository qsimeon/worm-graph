from pkg import *


@hydra.main(version_base=None, config_path="conf")
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

@hydra.main(version_base=None, config_path="conf", config_name="pipeline")
def my_app(cfg: DictConfig) -> None:
    """Create a custom pipeline"""

    print(OmegaConf.to_yaml(cfg), end="\n\n")

    # Verifications
    if len(cfg) == 0:
        raise ValueError("No submodules in the pipeline. Run python main.py +experiment=your_experiment")
    
    if 'train' in cfg.submodule:
        assert 'model' in cfg.submodule, "Model must be defined before training."
        assert 'dataset_train' in cfg.submodule, "Train dataset must be defined before training."

    if 'predict' in cfg.submodule:
        assert 'model' in cfg.submodule, "Model must be defined before making predictions."
        assert 'dataset_predict' in cfg.submodule, "Prediction dataset must be defined before making predictions."

    if 'visualize' in cfg.submodule:
        if cfg.submodule.visualize.log_dir is None:
             assert 'train' in cfg.submodule, "Train must be defined before visualizing (or chose a log_dir)."
             assert 'predict' in cfg.submodule, "Predict must be defined before visualizing (or chose a log_dir)."

    # Print all submodules to be included in the pipeline
    #print(OmegaConf.to_yaml(cfg.submodule))

    if 'preprocess' in cfg.submodule:
        process_data(cfg.submodule.preprocess) # working
    
    if 'dataset_train' in cfg.submodule:
        dataset_train = get_dataset(cfg.submodule.dataset_train) # working

    if 'dataset_predict' in cfg.submodule:
        dataset_predict = get_dataset(cfg.submodule.dataset_predict) # working

    if 'model' in cfg.submodule:
        model = get_model(cfg.submodule.model) # working

    if 'train' in cfg.submodule:
        model, submodules_updated, train_info = train_model(
            train_config = cfg.submodule.train,
            model = model,
            dataset = dataset_train
        ) # working
        # Update cfg.submodule parameters
        cfg.submodule = OmegaConf.merge(cfg.submodule, submodules_updated)

    if 'predict' in cfg.submodule:
        submodules_updated = make_predictions(
            predict_config = cfg.submodule.predict,
            model =  model,
            dataset = dataset_predict,
        ) # working
        # Update cfg.submodule parameters
        cfg.submodule = OmegaConf.merge(cfg.submodule, submodules_updated)

    # ================== Save updated configs ==================
    log_dir = os.getcwd()
    OmegaConf.save(cfg, os.path.join(log_dir, "pipeline_info.yaml"))

    if 'train' in cfg.submodule:
        OmegaConf.save(train_info, os.path.join(log_dir, "train_info.yaml"))
    # ==========================================================

    if 'visualize' in cfg.submodule:
        plot_figures(
            visualize_config=cfg.submodule.visualize
        )

    if 'analysis' in cfg.submodule:
        print(cfg.submodule.analysis)

if __name__ == "__main__":
    #! pipeline()
    my_app()
