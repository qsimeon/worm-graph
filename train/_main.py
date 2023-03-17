from train._utils import *


def train_model(
    model: torch.nn.Module,
    dataset: dict,
    config: DictConfig,
    optimizer: Union[torch.optim.Optimizer, None] = None,
    shuffle: bool = True,
) -> tuple[torch.nn.Module, str]:
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
        "logs", "{}-{}-{}".format(timestamp, dataset_name, model_class_name)
    )
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    # sample worms with replacement until desired number epochs (i.e. worms) obtained
    dataset_items = [
        (k, dataset[k])
        for k in np.random.choice(
            list(dataset.keys()), size=config.train.epochs, replace=True
        )
    ]
    # shuffle the dataset (without replacement)
    if shuffle == True:
        dataset_items = random.sample(dataset_items, k=len(dataset_items))
    # remake dataset with only selected worms
    dataset = dict(dataset_items)
    # instantiate the optimizer
    if optimizer is not None:
        optimizer = optimizer
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learn_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=config.train.learn_rate)
    # train/test loss metrics
    data = {
        "epochs": [],
        "base_train_losses": [],
        "base_test_losses": [],
        "train_losses": [],
        "test_losses": [],
        "num_train_samples": [],
        "num_test_samples": [],
        "centered_train_losses": [],
        "centered_test_losses": [],
    }
    # train the model for multiple cyles
    kwargs = dict(
        # args to `optimize_model`
        optimizer=optimizer,
        num_epochs=1,
        # args to `split_train_test`
        k_splits=config.train.k_splits,
        seq_len=config.train.seq_len,
        batch_size=config.train.batch_size,
        train_size=config.train.train_size,
        test_size=config.train.test_size,
        shuffle=config.train.shuffle,
        reverse=False,
        tau=1,  # deprecated
    )
    # choose whether to use original or smoothed calcium data
    if config.train.smooth_data:
        key_data = "smooth_calcium_data"
    else:
        key_data = "calcium_data"
    # train for config.train.num_epochs
    reset_epoch = 1
    for i, (worm, single_worm_dataset) in enumerate(dataset_items):
        # optimize for 1 epoch per (possibly duplicated) worm
        model, log = optimize_model(
            data=single_worm_dataset[key_data],
            model=model,
            mask=single_worm_dataset["named_neurons_mask"],
            start_epoch=reset_epoch,
            **kwargs,
        )
        # retrieve losses and sample counts
        [data[key].extend(log[key]) for key in data]
        # mutate the dataset for this worm with the train and test masks
        dataset[worm].setdefault("train_mask", log["train_mask"])
        dataset[worm].setdefault("test_mask", log["test_mask"])
        # set to next epoch
        reset_epoch = log["epochs"][-1] + 1
        if (i % config.train.save_freq == 0) or (i + 1 == config.train.epochs):
            # display progress
            print("num. worms trained on:", i + 1, "\nprevious worm:", worm, end="\n\n")
            # save model checkpoints
            chkpt_name = "{}_epochs_{}_worms.pt".format(reset_epoch - 1, i + 1)
            torch.save(
                {
                    "epoch": reset_epoch - 1,
                    "model_name": model_class_name,
                    "input_size": model.get_input_size(),
                    "hidden_size": model.get_hidden_size(),
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
    # make predictions with last saved model
    make_predictions(model, dataset, log_dir)
    # returned trained model and a path to log directory
    return model, log_dir


if __name__ == "__main__":
    config = OmegaConf.load("conf/train.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    model = get_model(OmegaConf.load("conf/model.yaml"))
    dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))
    model, log_dir = train_model(model, dataset, config)
