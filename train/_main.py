from train._utils import *


def train_model(
    model: torch.nn.Module,
    dataset: dict,
    config: DictConfig,
    shuffle: bool = True,  # whether to shuffle all worms
    log_dir: Union[str, None] = None,  # hydra passes
) -> tuple[torch.nn.Module, str]:
    """
    Trains a model on a multi-worm dataset. Returns the trained model
    and a path to the directory with training and evaluation logs.
    """
    # a worm dataset must have at least one worm: "worm0"
    assert "worm0" in dataset, "Not a valid dataset object."
    # get some helpful variables
    dataset_name = dataset["worm0"]["dataset"]
    model_class_name = model.__class__.__name__
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    num_unique_worms = len(dataset)
    # hydra changes the working directory to log directory
    if log_dir is None:
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
    # cycle the dataset until the desired number epochs obtained
    dataset_items = sorted(dataset.items()) * config.train.epochs
    assert (
        len(dataset_items) == config.train.epochs * num_unique_worms
    ), "Invalid number of worms."
    # shuffle (without replacement) the worms (including duplicates) in the dataset
    if shuffle == True:
        dataset_items = random.sample(dataset_items, k=len(dataset_items))
    # split the dataset into cohorts of worms; there should be one cohort per epoch
    worm_cohorts = np.array_split(dataset_items, config.train.epochs)
    worm_cohorts = [_.tolist() for _ in worm_cohorts]  # convert cohort arrays to lists
    assert len(worm_cohorts) == config.train.epochs, "Invalid number of cohorts."
    assert all(
        [len(cohort) == num_unique_worms for cohort in worm_cohorts]
    ), "Invalid cohort size."
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
    print("\nOptimizer:\n\t", optimizer, end="\n\n")
    # get other config params
    if config.get("globals"):
        use_residual = config.globals.use_residual
        smooth_data = config.train.smooth_data
    else:
        use_residual = False
        smooth_data = False
    # initialize train/test loss metrics arrays
    data = {
        "epochs": np.zeros(config.train.epochs, dtype=int),
        "num_train_samples": np.zeros(config.train.epochs, dtype=int),
        "num_test_samples": np.zeros(config.train.epochs, dtype=int),
        "base_train_losses": np.zeros(config.train.epochs, dtype=np.float32),
        "base_test_losses": np.zeros(config.train.epochs, dtype=np.float32),
        "train_losses": np.zeros(config.train.epochs, dtype=np.float32),
        "test_losses": np.zeros(config.train.epochs, dtype=np.float32),
        "centered_train_losses": np.zeros(config.train.epochs, dtype=np.float32),
        "centered_test_losses": np.zeros(config.train.epochs, dtype=np.float32),
    }
    # args to be passed to the `split_train_test` function
    kwargs = dict(
        k_splits=config.train.k_splits,
        seq_len=config.train.seq_len,
        batch_size=config.train.batch_size,
        train_size=config.train.train_size
        // num_unique_worms,  # keeps training set size constant per epoch (cohort)
        test_size=config.train.test_size
        // num_unique_worms,  # keeps validation set size constant per epoch (cohort)
        shuffle=config.train.shuffle,  # shuffle samples from each cohort
        reverse=config.train.reverse,  # generate samples backward from the end of data
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
    # throughout training, keep track of what neurons have been covered
    coverage_mask = torch.zeros_like(
        dataset["worm0"]["named_neurons_mask"], dtype=torch.bool
    )
    # memoize creation of data loaders and masks for speedup
    memo_loaders_masks = dict()
    # train for config.train.num_epochs
    reset_epoch = 0
    # reset_epoch = 1
    # main FOR loop; train the model for multiple epochs (one epoch per cohort)
    # for i, (worm, single_worm_dataset) in enumerate(dataset_items):
    for i, cohort in enumerate(worm_cohorts):
        # print("worms in cohort %s:" % i, [worm for worm, _ in cohort])  # DEBUGGING
        # create a list of loaders and masks for the cohort
        train_loader = np.empty(num_unique_worms, dtype=object)
        if i == 0:  # keep the validation dataset the same
            test_loader = np.empty(num_unique_worms, dtype=object)
        neurons_mask = np.empty(num_unique_worms, dtype=object)
        # iterate over each worm in the cohort
        for j, (worm, single_worm_dataset) in enumerate(cohort):
            # check the memo for existing loaders and masks
            if worm in memo_loaders_masks:
                train_loader[j] = memo_loaders_masks[worm]["train_loader"]
                if i == 0:  # keep the validation dataset the same
                    test_loader[j] = memo_loaders_masks[worm]["test_loader"]
                train_mask = memo_loaders_masks[worm]["train_mask"]
                test_mask = memo_loaders_masks[worm]["test_mask"]
            # create data loaders and train/test masks only once per worm
            else:
                (
                    train_loader[j],
                    tmp_test_loader,
                    train_mask,
                    test_mask,
                ) = split_train_test(
                    data=single_worm_dataset[key_data],
                    time_vec=single_worm_dataset.get(
                        "time_in_seconds", None
                    ),  # time vector
                    **kwargs,
                )
                if i == 0:  # keep the validation dataset the same
                    test_loader[j] = tmp_test_loader
                # add to memo
                memo_loaders_masks[worm] = dict(
                    train_loader=train_loader[j],
                    test_loader=tmp_test_loader,
                    train_mask=train_mask,
                    test_mask=test_mask,
                )
            # get the neurons mask for this worm
            neurons_mask[j] = single_worm_dataset["named_neurons_mask"]
            # update coverage mask
            coverage_mask |= single_worm_dataset["named_neurons_mask"]
            # mutate this worm's dataset with its train and test masks
            dataset[worm].setdefault("train_mask", train_mask.detach())
            dataset[worm].setdefault("test_mask", test_mask.detach())
        # create loaders and masks lists
        train_loader = list(train_loader)
        test_loader = list(test_loader)
        neurons_mask = list(neurons_mask)
        # optimize for 1 epoch per cohort
        num_epochs = 1
        # `optimize_model` can accept single loader/mask or list of loaders/masks
        model, log = optimize_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            neurons_mask=neurons_mask,
            optimizer=optimizer,
            start_epoch=reset_epoch,
            learn_rate=learn_rate,
            num_epochs=num_epochs,
            use_residual=use_residual,
        )
        # retrieve losses and sample counts
        for key in data:  # pre-allocated memory for `data[key]`
            # with `num_epochs=1`, the code below is just equal to data[key][i] = log[key]
            data[key][(i * num_epochs) : (i * num_epochs) + len(log[key])] = log[key]
        # get what neurons have been covered so far
        _ = torch.nonzero(coverage_mask).squeeze().numpy()
        covered_neurons = set(np.array(NEURONS_302)[_])
        # print(
        #     "cumulative number of neurons covered: %s" % len(covered_neurons)
        # )  # DEBUGGING
        # saving model checkpoints
        if (i % config.train.save_freq == 0) or (i + 1 == config.train.epochs):
            # display progress
            print(
                "Saving a model checkpoint.\n\tnum. worms trained on:",
                i + 1,
                "\n\tprevious worm:",
                worm,
                end="\n\n",
            )
            # save model checkpoints
            chkpt_name = "{}_epochs_{}_worms.pt".format(reset_epoch, i + 1)
            torch.save(
                {
                    "epoch": reset_epoch,
                    "dataset_name": dataset_name,
                    "model_name": model_class_name,
                    "timestamp": timestamp,
                    "covered_neurons": covered_neurons,
                    "input_size": model.get_input_size(),
                    "hidden_size": model.get_hidden_size(),
                    "num_cohorts": i + 1,
                    "num_worms": (i + 1) * num_unique_worms,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(log_dir, "checkpoints", chkpt_name),
            )
        # set to next epoch before continuing
        reset_epoch = log["epochs"][-1] + 1
        continue
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
        use_residual=use_residual,
        smooth_data=smooth_data,
    )
    # garbage collection
    gc.collect()
    # returned trained model and a path to log directory
    return model, log_dir


if __name__ == "__main__":
    config = OmegaConf.load("conf/train.yaml")
    print("\nconfig:\n\t", OmegaConf.to_yaml(config), end="\n\n")
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
