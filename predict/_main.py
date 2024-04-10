from predict._utils import *

# Init logger
logger = logging.getLogger(__name__)


def make_predictions(
    predict_config: DictConfig,
    model: torch.nn.Module,
) -> None:
    """Make predictions on a dataset with a trained model.

    Saves in log/predict a .csv with the Calcium neural activity predictions.

    Parameters:
    ----------
    predict_config : DictConfig
        Hydra configuration object.
    model : torch.nn.Module
        Trained model.
    train_dataset : torch.utils.data.Dataset
        Train dataset with worm data examples.
    val_dataset : dict
        Validation dataset with worm data examples.
    """
    # Use current working directory if one is not specified in predict_config
    if predict_config.predict_this_log_dir is None:
        log_dir = os.getcwd()  # logs/hydra/${now:%Y_%m_%d_%H_%M_%S}
    else:
        log_dir = predict_config.predict_this_log_dir
    # Get which datasets to predict
    source_datasets = predict_config.source_datasets
    # Set the context window if None
    context_window = predict_config.context_window
    if context_window is None:
        context_window = BLOCK_SIZE
        logger.info(f"Defaulting to context window: {context_window}\n")  # DEBUG
    # Create prediction directories
    os.makedirs(os.path.join(log_dir, "prediction"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "prediction", "train"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "prediction", "val"), exist_ok=True)
    # Make predictions in the train and validation datasets
    logger.info(f"Start making predictions.")
    model_predict(
        log_dir=log_dir,
        model=model,
        source_datasets=source_datasets,
        context_window=context_window,
    )
    return None


if __name__ == "__main__":
    # Get the necessary configs
    dataset_config = OmegaConf.load("configs/submodule/dataset.yaml")
    print(OmegaConf.to_yaml(dataset_config), end="\n\n")
    model_config = OmegaConf.load("configs/submodule/model.yaml")
    print(OmegaConf.to_yaml(model_config), end="\n\n")
    predict_config = OmegaConf.load("configs/submodule/predict.yaml")
    print(OmegaConf.to_yaml(predict_config), end="\n\n")
    # Create new to log directory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join("logs/hydra", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    # Move to new log directory
    os.chdir(log_dir)
    # Get the model
    model = get_model(model_config.model)
    # Get the dataset
    dataset = get_datasets(dataset_config.dataset)
    # Make predictions
    make_predictions(
        model=model,
        dataset=dataset,
        predict_config=predict_config.predict,
    )
