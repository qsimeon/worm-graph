from train._utils import *


def train_model(
    model: torch.nn.Module,
    dataset: dict,
    config: DictConfig,
    shuffle: bool = True,  # whether to shuffle worms
    log_dir: Union[str, None] = None,  # hydra passes
) -> tuple[torch.nn.Module, str]:
    """
    Trains a model on a multi-worm dataset. Returns the trained model
    and a path to the directory with training and evaluation logs.
    """
    assert "worm0" in dataset, "Not a valid dataset object."
    # initialize
    num_unique_worms = len(dataset)
    dataset_name = dataset["worm0"]["dataset"]
    model_class_name = model.__class__.__name__
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if log_dir is None:  # hydra changes working directory to log directory
        log_dir = os.getcwd()
    os.makedirs(log_dir, exist_ok=True)
    # create a model checkpoints folder
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    # modify the config file
    config = OmegaConf.structured(OmegaConf.to_yaml(config))
    config.setdefault("dataset", {"name": dataset_name})
    config.setdefault("model", {"type": model_class_name})
    config.setdefault("timestamp", timestamp)
    config.setdefault("num_unique_worms", num_unique_worms)
    # save config to log directory
    OmegaConf.save(config, os.path.join(log_dir, "config.yaml"))
    # cycle the dataset until the desired number epochs (i.e. worms) obtained
    # dataset_items = (
    #     sorted(dataset.items()) * (config.train.epochs // num_unique_worms)
    #     + sorted(dataset.items())[: (config.train.epochs % num_unique_worms)]
    # )
    # # remake dataset with only selected worms
    # dataset = dict(dataset_items)
    dataset_items = sorted(dataset.items()) * config.train.epochs
    # shuffle the worms in the dataset (without replacement)
    if shuffle == True:
        dataset_items = random.sample(dataset_items, k=len(dataset_items))
    # split the worms into cohorts per epoch
    worm_cohorts = np.array_split(dataset_items, config.train.epochs)
    worm_cohorts = [_.tolist() for _ in worm_cohorts]
    # instantiate the optimizer
    opt_param = config.train.optimizer
    learn_rate = config.train.learn_rate
    if config.train.optimizer is not None:
        if isinstance(opt_param, str):
            optimizer = eval(
                "torch.optim."
                + opt_param
                + "(model.parameters(), lr="
                + str(learn_rate)
                + ")"
            )
        assert isinstance(
            optimizer, torch.optim.Optimizer
        ), "Please use an instance of torch.optim.Optimizer."
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    print("Optimizer:", optimizer, end="\n\n")
    # get other config params
    if config.get("globals"):
        use_residual = config.globals.use_residual
        smooth_data = config.train.smooth_data
    else:
        use_residual = False
        smooth_data = False
    # train/test loss metrics
    data = {
        "epochs": np.zeros(len(dataset_items), dtype=int),
        "num_train_samples": np.zeros(len(dataset_items), dtype=int),
        "num_test_samples": np.zeros(len(dataset_items), dtype=int),
        "base_train_losses": np.zeros(len(dataset_items), dtype=np.float32),
        "base_test_losses": np.zeros(len(dataset_items), dtype=np.float32),
        "train_losses": np.zeros(len(dataset_items), dtype=np.float32),
        "test_losses": np.zeros(len(dataset_items), dtype=np.float32),
        "centered_train_losses": np.zeros(len(dataset_items), dtype=np.float32),
        "centered_test_losses": np.zeros(len(dataset_items), dtype=np.float32),
    }
    # train the model for multiple cyles
    kwargs = dict(  # args to `split_train_test`
        k_splits=config.train.k_splits,
        seq_len=config.train.seq_len,
        batch_size=config.train.batch_size,
        # train_size=config.train.train_size,
        # test_size=config.train.test_size,
        train_size=config.train.train_size // num_unique_worms,
        test_size=config.train.test_size // num_unique_worms,
        shuffle=config.train.shuffle,  # whether to shuffle the samples from a worm
        reverse=False,
        tau=config.train.tau_in,
        use_residual=use_residual,
    )
    # choose whether to use calcium or residual data
    if use_residual:
        key_data = "residual_calcium"
    else:
        key_data = "calcium_data"
    # choose whether to use original or smoothed data
    if smooth_data:
        key_data = "smooth_" + key_data
    else:
        key_data = key_data
    # memoize creation of data loaders and masks for speedup
    memo_loaders_masks = dict()
    # train for config.train.num_epochs
    reset_epoch = 1
    # main FOR loop
    # for i, (worm, single_worm_dataset) in enumerate(dataset_items):
    for i, cohort in enumerate(worm_cohorts):
        # create a list of loaders and masks for the cohort
        train_loaders = []
        test_loaders = []
        neurons_masks = []
        # iterate over each worm in the cohort
        for worm, single_worm_dataset in cohort:
            # check the memo for existing loaders and masks
            if worm in memo_loaders_masks:
                train_loader = memo_loaders_masks[worm]["train_loader"]
                test_loader = memo_loaders_masks[worm]["test_loader"]
                train_mask = memo_loaders_masks[worm]["train_mask"]
                test_mask = memo_loaders_masks[worm]["test_mask"]
            # create data loaders and train/test masks only once per worm
            else:
                train_loader, test_loader, train_mask, test_mask = split_train_test(
                    data=single_worm_dataset[key_data],
                    time_vec=single_worm_dataset.get(
                        "time_in_seconds", None
                    ),  # time vector
                    **kwargs,
                )
                # add to memo
                memo_loaders_masks[worm] = dict(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    train_mask=train_mask,
                    test_mask=test_mask,
                )
            # insert train and test masks for worm into its dataset
            dataset[worm].setdefault("train_mask", train_mask.detach())
            dataset[worm].setdefault("test_mask", test_mask.detach())
            # get the neurons mask for this worm
            neurons_mask = single_worm_dataset["named_neurons_mask"]
            # add to the list of loaders and masks
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)
            neurons_masks.append(neurons_mask)
        # optimize for 1 epoch per cohort
        num_epochs = 1
        model, log = optimize_model(
            model=model,
            train_loader=train_loaders,  # list of loaders
            test_loader=test_loaders,  # list of loaders
            neurons_mask=neurons_masks,  # list of masks
            optimizer=optimizer,
            start_epoch=reset_epoch,
            learn_rate=learn_rate,
            num_epochs=num_epochs,
            use_residual=use_residual,
        )
        # retrieve losses and sample counts
        for key in data:  # pre-allocated memory for data[key]
            data[key][(i * num_epochs) : (i * num_epochs) + len(log[key])] = log[key]
        # set to next epoch
        reset_epoch = log["epochs"][-1] + 1
        # outputs
        if (i % config.train.save_freq == 0) or (i + 1 == config.train.epochs):
            # display progress
            print("num. worms trained on:", i + 1, "\nprevious worm:", worm, end="\n\n")
            # save model checkpoints
            chkpt_name = "{}_epochs_{}_worms.pt".format(reset_epoch - 1, i + 1)
            torch.save(
                {
                    "epoch": reset_epoch - 1,
                    "dataset_name": dataset_name,
                    "model_name": model_class_name,
                    "timestamp": timestamp,
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
    make_predictions(
        model,
        dataset,
        log_dir,
        tau=config.train.tau_out,
        # tau=config.train.tau_in,
        use_residual=use_residual,
        smooth_data=smooth_data,
    )
    # garbage collection
    gc.collect()
    # returned trained model and a path to log directory
    return model, log_dir


if __name__ == "__main__":
    config = OmegaConf.load("conf/train.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    model = get_model(OmegaConf.load("conf/model.yaml"))
    dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model, log_dir = train_model(
        model,
        dataset,
        config,
        shuffle=config.train.shuffle,
        log_dir=os.path.join("logs", "{}".format(timestamp)),
    )
