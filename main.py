from pkg import *

@hydra.main(version_base=None, config_path="conf", config_name="pipeline")
def pipeline(cfg: DictConfig) -> None:
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

    torch_device() # Display Pytorch device

    init_random_seeds(cfg.seed) # Intialize random seeds

    if 'preprocess' in cfg.submodule:
        process_data(cfg.submodule.preprocess)
    
    if 'dataset_train' in cfg.submodule:
        dataset_train = get_dataset(cfg.submodule.dataset_train)

    if 'dataset_predict' in cfg.submodule:
        dataset_predict = get_dataset(cfg.submodule.dataset_predict)

    if 'model' in cfg.submodule:
        model = get_model(cfg.submodule.model)

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

    clear_cache() # Free up GPU

if __name__ == "__main__":
    pipeline()
