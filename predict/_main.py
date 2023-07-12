from predict._utils import *


def make_predictions(
    config: DictConfig,
    model: Union[torch.nn.Module, None] = None,
    dataset: Union[dict, None] = None,
    log_dir: Union[str, None] = None,  # hydra passes this in
    use_residual: bool = False,
    smooth_data: bool = True,
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

    # Hydra changes the working directory to log directory (logs/playground)
    if log_dir is None:
        log_dir = os.getcwd()
    else:
        os.makedirs(log_dir, exist_ok=True)

    # Replace model and dataset with those in config file
    if model is None:
        model = get_model(config.predict)
    if dataset is None:
        dataset = get_dataset(config.predict)

    # Worms to exclude
    leave_out_worms = list(config.predict.dataset.leave_out_worms)

    # Get the desired number of future timesteps to predict
    future_timesteps = config.predict.tau_out

    # Get the desired context window length
    context_len = config.predict.context_len

    # Get the data to save
    signal_str = "residual" if use_residual else "calcium"
    key_data = "residual_calcium" if use_residual else "calcium_data"
    key_data = "smooth_" + key_data if smooth_data else key_data

    # Make a directory for each worm in the dataset and store the data
    for worm, single_worm_dataset in dataset.items():
        # skip excluded worms
        if worm in leave_out_worms:
            continue

        # make a directory for each worm
        os.makedirs(os.path.join(log_dir, worm), exist_ok=True)

        # Get data to save
        calcium_data = single_worm_dataset[key_data]

        # Pick just named neurons
        named_neurons_mask = single_worm_dataset["named_neurons_mask"]
        named_neurons = np.array(NEURONS_302)[named_neurons_mask]

        time_in_seconds = single_worm_dataset["time_in_seconds"]
        train_mask = single_worm_dataset.setdefault(
            "train_mask", torch.zeros(len(calcium_data), dtype=torch.bool)
        )
        test_mask = single_worm_dataset.setdefault("test_mask", ~train_mask)

        # Detach computation from tensors
        calcium_data = calcium_data.detach()
        named_neurons_mask = named_neurons_mask.detach()
        time_in_seconds = time_in_seconds.detach()
        train_mask = train_mask.detach()
        test_mask.detach()

        # shorten time series to the maximum token lengths
        calcium_data = calcium_data[-MAX_TOKEN_LEN:, :]
        time_in_seconds = time_in_seconds[-MAX_TOKEN_LEN:]
        train_mask = train_mask[-MAX_TOKEN_LEN:]
        test_mask = test_mask[-MAX_TOKEN_LEN:]

        # Labels and columns
        train_labels = np.expand_dims(np.where(train_mask, "train", "test"), axis=-1)
        columns = list(named_neurons) + [
            "train_test_label",
            "time_in_seconds",
            "tau",
        ]

        # Put model and data on device
        model = model.to(DEVICE)
        calcium_data = calcium_data.to(DEVICE)
        named_neurons_mask = named_neurons_mask.to(DEVICE)

        # Make predictions with final model
        predictions, targets = model_predict(
            model,
            data=calcium_data,
            timesteps=future_timesteps,
            context_window=context_len,
            mask=named_neurons_mask,
        )

        # Save Calcium DataFrame
        tau_expand = np.full(time_in_seconds.shape, future_timesteps)
        data = calcium_data[:, named_neurons_mask].cpu().numpy()
        data = np.hstack((data, train_labels, time_in_seconds, tau_expand))
        pd.DataFrame(data=data, columns=columns).to_csv(
            os.path.join(log_dir, worm, signal_str + "_activity.csv"),
            index=True,
            header=True,
        )

        # Save Predictions DataFrame
        data = predictions[:, named_neurons_mask].cpu().numpy()
        data = np.hstack((data, train_labels, time_in_seconds, tau_expand))
        pd.DataFrame(data=data, columns=columns).to_csv(
            os.path.join(log_dir, worm, "predicted_" + signal_str + ".csv"),
            index=True,
            header=True,
        )

        # Save Targets DataFrame
        data = targets[:, named_neurons_mask].cpu().numpy()
        data = np.hstack((data, train_labels, time_in_seconds, tau_expand))
        pd.DataFrame(data=data, columns=columns).to_csv(
            os.path.join(log_dir, worm, "target_" + signal_str + ".csv"),
            index=True,
            header=True,
        )

    return None


if __name__ == "__main__":
    config = OmegaConf.load("conf/predict.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    if config.predict.model.checkpoint_path.__contains__("checkpoints"):
        log_dir = config.predict.model.checkpoint_path.split("checkpoints")[0]
    else:
        log_dir = os.path.join("logs", "playground")
    make_predictions(
        config,
        log_dir=log_dir,
    )
