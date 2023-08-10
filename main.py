from pkg import *

@hydra.main(version_base=None, config_path="configs", config_name="pipeline")
def pipeline(cfg: DictConfig) -> None:
    """Create a custom pipeline"""

    # Configure logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Start new experiment run (MLflow)
    mlflow.set_tracking_uri(LOGS_DIR+'/mlruns')
    mlflow.set_experiment(cfg.experiment.name)

    with mlflow.start_run(run_name=datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) as run:

        # Verifications
        if len(cfg) == 1: # only the pipeline module
            raise ValueError("No submodules in the pipeline. Run python main.py +experiment=your_experiment")

        if 'visualize' in cfg.submodule:
            if cfg.submodule.visualize.log_dir is None:
                assert ('train' in cfg.submodule) or ('predict' in cfg.submodule), "Train/Predict must be defined before visualizing (or chose a log_dir)."

        logger.info("Torch device: %s" % (DEVICE))

        torch.cuda.empty_cache()

        if cfg.experiment.seed is None:
            cfg.experiment.seed = random.randint(0, 100)
        logger.info("Setting random seeds to %d" % (cfg.experiment.seed))
        init_random_seeds(cfg.experiment.seed) # Set random seeds

        if 'preprocess' in cfg.submodule:
            process_data(cfg.submodule.preprocess)
        
        if 'dataset' in cfg.submodule:
            if 'for_training' in cfg.submodule.dataset:
                dataset_train = get_datasets(cfg.submodule.dataset.for_training, name='training')
            if 'for_prediction' in cfg.submodule.dataset:
                dataset_predict = get_datasets(cfg.submodule.dataset.for_prediction, name='prediction')
        else:
            dataset_train = (None, None)
            dataset_predict = (None, None)

        if 'model' in cfg.submodule:
            model = get_model(cfg.submodule.model)
        else:
            model = None

        if 'train' in cfg.submodule:
            model, metric = train_model(
                train_config = cfg.submodule.train,
                model = model,
                train_dataset = dataset_train[0],
                val_dataset = dataset_train[1],
                verbose = False
            )

        if 'predict' in cfg.submodule:
            submodules_updated = make_predictions(
                predict_config = cfg.submodule.predict,
                model =  model,
                dataset = dataset_predict,
            )
            # Update cfg.submodule
            cfg.submodule.dataset = OmegaConf.merge(cfg.submodule.dataset, submodules_updated.dataset) # update dataset.predict name
            if 'visualize' in cfg.submodule:
                cfg.submodule.visualize = OmegaConf.merge(cfg.submodule.visualize, submodules_updated.visualize) # update log_dir

        # ================== Save updated configs ==================

        log_dir = os.getcwd()

        OmegaConf.save(cfg, os.path.join(log_dir, "pipeline_info.yaml"))
        
        # ==========================================================

        if 'visualize' in cfg.submodule:
            plot_figures(
                visualize_config = cfg.submodule.visualize,
            )

        if 'analysis' in cfg.submodule:
            pass

        if cfg.experiment.name in ['num_worms', 'num_named_neurons']:
            log_dir = os.path.dirname(log_dir) # Because experiments run in multirun mode
            plot_experiment(log_dir, cfg.experiment)
        
        # Save experiment parameters (MLflow)
        log_params_from_omegaconf_dict(cfg)

        torch.cuda.empty_cache()
        mlflow.end_run()

    # Return metric for optuna automatic hyperparameter tuning
    if 'train' in cfg.submodule:
        logger.info("Experiment finished. Final metric: %s" % (metric))
        return metric

if __name__ == "__main__":
    pipeline()