from train._utils import *


def train_model(
    model: torch.nn.Module, dataset: dict, config: DictConfig, optimizer=None
):
    """
    Trains the
    """
    assert ("name" in dataset) and (
        "generator" in dataset
    ), "Not a valid dataset object."
    # initialize
    logs = dict(
        dataset_name=dataset["name"],
        model_class_name=model.__class__.__name__,
        timestamp=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
    )
    log_dir = os.path.join(
        LOGS_DIR,
        "{}-{}-{}".format(logs["dataset_name"], logs["model_name"], logs["timestamp"]),
    )
    os.makedirs(path=log_dir, exist_ok=True)
    data_gen = dataset["generator"]
    # instantiate the optimizer
    if optimizer is not None:
        optimizer = optimizer
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learn_rate)
    # train the model
    epoch = 1
    for worm, single_worm_dataset in data_gen:
        model, log = optimize_model(
            dataset=single_worm_dataset["calcium_data"],
            model=model,
            mask=single_worm_dataset["named_neurons_mask"],
            optimizer=optimizer,
            start_epoch=epoch,
            num_epochs=config.train.epochs,
            seq_len=config.train.seq_len,
            dataset_size=config.train.dataset_size,
        )
        logs[worm] = log
        epoch = log["epochs"][-1] + 1
    # make predicitons with trained model and save logs
    for worm, single_worm_dataset in data_gen:
        targets, predictions = model_predict(single_worm_dataset["calcium_data"], model)
        logs[worm].update(
            {
                "neurons_mask": single_worm_dataset["neurons_mask"],
                "calcium_data": single_worm_dataset["calcium_data"],
                "slot_to_neuron": single_worm_dataset["slot_to_neuron"],
                "target_ca_residual": targets,
                "predicted_ca_residual": predictions,
            }
        )
        # save logs as Pandas dataframes
        os.makedirs(path=os.path.join(log_dir, worm), exist_ok=True)
        # df = pd.DataFrame(, columns=["neuron", ], index=logs[worm]["slot_to_neuron"].keys())
        # df.to_csv(file, mode='a', index=True, columns=columns, header=header)

    # returned trained model and path to log directory
    return model, log_dir


if __name__ == "__main__":
    model = get_model(OmegaConf.load("conf/model.yaml"))
    dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))
    config = OmegaConf.load("conf/train.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    model, log_dir = train_model(model, dataset, config)
