from predict._utils import *

# Init logger
logger = logging.getLogger(__name__)

def make_predictions(
    predict_config: DictConfig,
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
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
    for ds_type in os.listdir(os.path.join(log_dir, 'prediction')):

        if ds_type not in ['train', 'val']:
            continue

        logger.info("Start making predictions in %s dataset..." % (ds_type))

        model_predict(
            log_dir = os.path.join(log_dir, 'prediction', ds_type),
            model = model,
            dataset = train_dataset if ds_type == 'train' else val_dataset,
            context_window = predict_config.context_window,
            nb_ts_to_generate = predict_config.nb_ts_to_generate,
            worms_to_predict = predict_config.worms_to_predict,
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