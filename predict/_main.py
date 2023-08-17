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

    Saves in the provided log directory a .csv file for each of the
    following:
        * Calcium neural activity
        * Target calcium residuals
        * Predicted calcium residuals
    Each .csv file has a column for each named neuron in the dataset
    plus two additional columns for the train mask and test mask
    respectively.

    Parameters:
    ----------
    config : DictConfig
        Hydra configuration object.
    model : torch.nn.Module or None, optional, default: None
        Trained model. If not provided, will be loaded from the configuration.
    dataset : dict or None, optional, default: None
        Multi-worm dataset. If not provided, will be loaded from the configuration.
    log_dir : str or None, optional, default: None
        Log directory where the output files will be saved (logs/playground).
    use_residual : bool, optional, default: False
        If True, use residual calcium instead of calcium data.
    smooth_data : bool, optional, default: True
        If True, use smoothed data for predictions.

    Calls
    -----
    get_model : function in model/_main.py
        Instantiate or load a model as specified in 'model.yaml'.
    get_dataset : function in data/_main.py
        Returns a dict with the worm data of all requested datasets.
    model_predict : function in predict/_utils.py
        Make predictions for all neurons on a dataset with a trained model.

    Returns
    -------
    None

    Notes
    -----
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