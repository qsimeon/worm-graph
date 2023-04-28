from predict._utils import *


def make_predictions(
    config: DictConfig,
    model: Union[torch.nn.Module, None] = None,
    dataset: Union[dict, None] = None,
    log_dir: Union[str, None] = None,  # hydra passes this in
    use_residual: bool = False,
    smooth_data: bool = True,
) -> None:
    """Make predicitons on a dataset with a trained model.

    Saves in the provdied log directory a .csv file for each of the following:
        * calcium neural activty
        * target calcium residuals
        * predicted calcium residuals
    Each .csv file has a column for each named neuron in the dataset plus two
    additional columns for the train mask and test mask respectively.

    Args:
        model: torch.nn.Module, Trained model.
        dataset: dict, Multi-worm dataset.

    Returns:
        None.
    """
    # hydra changes the working directory to log directory
    if log_dir is None:
        log_dir = os.getcwd()
    print(log_dir)
    # replace model and dataset with those in config file
    if model is None:
        model = get_model(config.predict)
    if dataset is None:
        dataset = get_dataset(config.predict)
    # get the desired output shift
    tau = config.predict.tau_out
    # get the data to save
    signal_str = "residual" if use_residual else "calcium"
    key_data = "residual_calcium" if use_residual else "calcium_data"
    key_data = "smooth_" + key_data if smooth_data else key_data
    # make a directory for each worm in the dataset and store the data
    for worm, single_worm_dataset in dataset.items():
        os.makedirs(os.path.join(log_dir, worm), exist_ok=True)
        # get data to save
        calcium_data = single_worm_dataset[key_data]
        named_neurons_mask = single_worm_dataset["named_neurons_mask"]
        named_neurons = np.array(NEURONS_302)[named_neurons_mask]
        time_in_seconds = single_worm_dataset["time_in_seconds"]
        if time_in_seconds is None:
            time_in_seconds = torch.arange(len(calcium_data)).double()
        train_mask = single_worm_dataset.setdefault(
            "train_mask", torch.zeros(len(calcium_data), dtype=torch.bool)
        )
        test_mask = single_worm_dataset.setdefault("test_mask", ~train_mask)
        # detach computation from tensors
        calcium_data = calcium_data.detach()
        named_neurons_mask = named_neurons_mask.detach()
        time_in_seconds = time_in_seconds.detach()
        train_mask = train_mask.detach()
        test_mask.detach()
        # labels and columns
        labels = np.expand_dims(np.where(train_mask, "train", "test"), axis=-1)
        columns = list(named_neurons) + [
            "train_test_label",
            "time_in_seconds",
            "tau",
        ]
        # make predictions with final model
        targets, predictions = model_predict(
            model,
            calcium_data * named_neurons_mask,
            tau=tau,
        )
        # save dataframes
        tau_expand = np.full(time_in_seconds.shape, tau)
        data = calcium_data[:, named_neurons_mask].numpy()
        data = np.hstack((data, labels, time_in_seconds, tau_expand))
        pd.DataFrame(data=data, columns=columns).to_csv(
            os.path.join(log_dir, worm, signal_str + "_activity.csv"),
            index=True,
            header=True,
        )
        data = targets[:, named_neurons_mask].detach().numpy()
        data = np.hstack((data, labels, time_in_seconds, tau_expand))
        pd.DataFrame(data=data, columns=columns).to_csv(
            os.path.join(log_dir, worm, "target_" + signal_str + ".csv"),
            index=True,
            header=True,
        )
        columns = list(named_neurons) + [
            "train_test_label",
            "time_in_seconds",
            "tau",
        ]
        data = predictions[:, named_neurons_mask].detach().numpy()
        data = np.hstack((data, labels, time_in_seconds, tau_expand))
        pd.DataFrame(data=data, columns=columns).to_csv(
            os.path.join(log_dir, worm, "predicted_" + signal_str + ".csv"),
            index=True,
            header=True,
        )
        # TODO: remove this break to do predict for all worms in dataset
        break
    return None


if __name__ == "__main__":
    config = OmegaConf.load("conf/predict.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # make predictions on all worms in the given dataset with a saved model
    make_predictions(
        config,
        log_dir=os.path.join("logs", "{}".format(timestamp)),
    )
