from train._utils import *


def train_model(
    model: torch.nn.Module,
    dataset: dict,
    config: DictConfig,
    optimizer=None,
    shuffle=True,
):
    """
    Trains a model on a multi-worm dataset. Returns the trained model
    and a path to the directory with training and evaluation logs.
    """
    assert "worm0" in dataset, "Not a valid dataset object."
    # initialize
    dataset_name = dataset["worm0"]["dataset"]
    model_class_name = model.__class__.__name__
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    log_dir = os.path.join(
        LOGS_DIR, "{}-{}-{}".format(dataset_name, model_class_name, timestamp)
    )
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    # whether to shuffle the dataset
    if shuffle == True:
        dataset_items = random.sample(list(dataset.items()), k=len(dataset))
    else:
        dataset_items = dataset.items()
    # instantiate the optimizer
    if optimizer is not None:
        optimizer = optimizer
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learn_rate)
    # train/test metrics
    data = {
        "epochs": [],
        "base_train_losses": [],
        "base_test_losses": [],
        "train_losses": [],
        "test_losses": [],
        "centered_train_losses": [],
        "centered_test_losses": [],
    }
    # train the model
    reset_epoch = 1
    for i, (worm, single_worm_dataset) in enumerate(dataset_items):
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
        print("num. worms trained on:", i + 1, "\nprevious worm:", worm, end="\n\n")
        # retrieve losses
        data["epochs"].extend(log["epochs"])
        data["train_losses"].extend(log["train_losses"])
        data["test_losses"].extend(log["test_losses"])
        data["base_train_losses"].extend(log["base_train_losses"])
        data["base_test_losses"].extend(log["base_test_losses"])
        data["centered_train_losses"].extend(log["centered_train_losses"])
        data["centered_test_losses"].extend(log["centered_test_losses"])
        reset_epoch = log["epochs"][-1] + 1
        # save model checkpoints
        chkpt_name = "{}_epochs_{}_worms.pt".format(reset_epoch - 1, i + 1)
        torch.save(
            {
                "epoch": reset_epoch - 1,
                "num_worms": i + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(log_dir, "checkpoints", chkpt_name),
        )
    # save loss curves
    pd.DataFrame(data=data).to_csv(
        os.path.join(log_dir, "loss_curves.csv"),
        index=True,
        header=True,
    )
    # make predictions with final trained model
    make_predictions(model, dataset, log_dir)
    # returned trained model and path to log directory
    return model, log_dir


if __name__ == "__main__":
    config = OmegaConf.load("conf/train.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    model = get_model(OmegaConf.load("conf/model.yaml"))
    dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))
    model, log_dir = train_model(model, dataset, config)
