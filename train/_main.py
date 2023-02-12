from train._utils import *


def train_model(
    model: torch.nn.Module,
    dataset: dict,
    config: DictConfig,
    optimizer=None,
):
    """
    Trains a model on a multi-worm dataset. Returns the trained model
    and a path to the directory with training and evaluation logs.
    """
    assert "worm0" in dataset, "Not a valid dataset object."
    # initialize
    logs = dict()
    dataset_name = dataset["worm0"]["dataset"]
    model_class_name = model.__class__.__name__
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join(
        LOGS_DIR, "{}-{}-{}".format(dataset_name, model_class_name, timestamp)
    )
    os.makedirs(log_dir, exist_ok=True)
    # instantiate the optimizer
    if optimizer is not None:
        optimizer = optimizer
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learn_rate)
    # train the model and accumulate the log metrics
    columns = [
        "epochs",
        "base_train_losses",
        "base_test_losses",
        "train_losses",
        "test_losses",
    ]
    data = {
        "worms_trained_on": [],
        "num_worms_trained_on": [],
        "epochs": [],
        "base_train_losses": [],
        "base_test_losses": [],
        "train_losses": [],
        "test_losses": [],
    }
    reset_epoch = 1
    for worm, single_worm_dataset in dataset.items():
        model, log = optimize_model(
            dataset=single_worm_dataset["calcium_data"],
            model=model,
            mask=single_worm_dataset["named_neurons_mask"],
            optimizer=optimizer,
            start_epoch=reset_epoch,
            num_epochs=config.train.epochs,
            seq_len=config.train.seq_len,
            dataset_size=config.train.dataset_size,
        )
        logs[worm] = log
        reset_epoch = log["epochs"][-1] + 1
    # TODO: put this second loop in a helper function
    # make predicitons with trained model and save logs
    for worm, single_worm_dataset in dataset.items():
        os.makedirs(os.path.join(log_dir, worm), exist_ok=True)
        # make predictions with final model
        targets, predictions = model_predict(single_worm_dataset["calcium_data"], model)
        # get data to save
        named_neuron_to_idx = single_worm_dataset["named_neuron_to_idx"]
        calcium_data = single_worm_dataset["calcium_data"]
        named_neurons_mask = single_worm_dataset["named_neurons_mask"]
        # save training curves

        # save dataframes and model checkpoints
        columns = list(named_neuron_to_idx)
        data = calcium_data[:, named_neurons_mask].numpy()
        pd.DataFrame(data=data, columns=columns).to_csv(
            os.path.join(log_dir, worm, "ca_activity.csv"),
            index=True,
            header=True,
        )
        data = torch.nn.functional.pad(
            targets[:, named_neurons_mask], (0, 0, 1, 0)
        ).numpy()
        pd.DataFrame(data=data, columns=columns).to_csv(
            os.path.join(log_dir, worm, "target_ca_residual.csv"),
            index=True,
            header=True,
        )
        data = torch.nn.functional.pad(
            predictions[:, named_neurons_mask], (0, 0, 0, 1)
        ).numpy()
        pd.DataFrame(data=data, columns=columns).to_csv(
            os.path.join(log_dir, worm, "predicted_ca_residual.csv"),
            index=True,
            header=True,
        )

    # returned trained model and path to log directory
    return model, log_dir


if __name__ == "__main__":
    model = get_model(OmegaConf.load("conf/model.yaml"))
    dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))
    config = OmegaConf.load("conf/train.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    model, log_dir = train_model(model, dataset, config)
