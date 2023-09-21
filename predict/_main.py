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

    log_dir = os.getcwd()  # logs/hydra/${now:%Y_%m_%d_%H_%M_%S}

    # Create prediction directories
    os.makedirs(os.path.join(log_dir, 'prediction'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'prediction', 'train'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'prediction', 'val'), exist_ok=True)

    # Make predictions in the train and validation datasets
    logger.info("Start making predictions.")
    model_predict(
        log_dir = log_dir,
        model = model,
        experimental_datasets = predict_config.experimental_datasets,
        context_window = predict_config.context_window,
    )

if __name__ == "__main__":
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
    dataset = get_datasets(dataset_config.dataset.predict)

    # Make predictions
    make_predictions(
        model=model,
        dataset=dataset,
        predict_config=predict_config.predict,
    )