from train._utils import *


def train_model(
    config: DictConfig,
    model: torch.nn.Module,
    dataset: dict,
    shuffle: bool = True,  # whether to shuffle all worms
    log_dir: Union[str, None] = None,  # hydra passes this in
) -> tuple[torch.nn.Module, str]:
    """
    Trains a model on a multi-worm dataset. Returns the trained model
    and a path to the directory with training and evaluation logs.
    """
    # a valid worm dataset must have at least one worm: "worm0"
    assert "worm0" in dataset, "Not a valid dataset object."
    # get some helpful variables
    dataset_name = dataset["worm0"]["dataset"]
    model_class_name = model.__class__.__name__
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    num_unique_worms = len(dataset)
    train_epochs = config.train.epochs + 1
    shuffle_worms = shuffle
    shuffle_sequences = config.train.shuffle
    batch_size = config.train.num_samples // config.train.num_batches
    # hydra changes the working directory to log directory
    if log_dir is None:
        log_dir = os.getcwd()
    os.makedirs(log_dir, exist_ok=True)
    # create a model checkpoints folder
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    # cycle the dataset until the desired number epochs obtained
    dataset_items = sorted(dataset.items()) * train_epochs
    assert (
        len(dataset_items) == train_epochs * num_unique_worms
    ), "Invalid number of worms."
    # shuffle (without replacement) the worms (including duplicates) in the dataset
    if shuffle_worms == True:
        dataset_items = random.sample(dataset_items, k=len(dataset_items))
    # split the dataset into cohorts of worms; there should be one cohort per epoch
    worm_cohorts = np.array_split(dataset_items, train_epochs)
    worm_cohorts = [_.tolist() for _ in worm_cohorts]  # convert cohort arrays to lists
    assert len(worm_cohorts) == train_epochs, "Invalid number of cohorts."
    assert all(
        [len(cohort) == num_unique_worms for cohort in worm_cohorts]
    ), "Invalid cohort size."
    # instantiate the optimizer
    opt_param = config.train.optimizer
    optim_name = "torch.optim." + opt_param
    learn_rate = config.train.learn_rate
    if config.train.optimizer is not None:
        if isinstance(opt_param, str):
            optimizer = eval(
                optim_name + "(model.parameters(), lr=" + str(learn_rate) + ")"
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
        smooth_data = config.globals.smooth_data
    else:
        use_residual = False
        smooth_data = False
    # initialize train/test loss metrics arrays
    data = {
        "epochs": np.zeros(train_epochs, dtype=int),
        "num_train_samples": np.zeros(train_epochs, dtype=int),
        "num_test_samples": np.zeros(train_epochs, dtype=int),
        "base_train_losses": np.zeros(train_epochs, dtype=np.float32),
        "base_test_losses": np.zeros(train_epochs, dtype=np.float32),
        "train_losses": np.zeros(train_epochs, dtype=np.float32),
        "test_losses": np.zeros(train_epochs, dtype=np.float32),
        "centered_train_losses": np.zeros(train_epochs, dtype=np.float32),
        "centered_test_losses": np.zeros(train_epochs, dtype=np.float32),
    }
    # create keyword arguments for `split_train_test`
    kwargs = dict(
        k_splits=config.train.k_splits,
        seq_len=config.train.seq_len,
        num_samples=config.train.num_samples,  # number of samples per worm
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
    # keep track of the amount of train data per epoch (in timesteps)
    worm_timesteps = 0
    # keep track of the total optimization time
    cumsum_seconds = 0
    # keep track of the average optimization time per epoch
    seconds_per_epoch = 0
    # main FOR loop; train the model for multiple epochs (one cohort epoch)
    for i, cohort in enumerate(worm_cohorts):
        # create a array of datasets and masks for the cohort
        train_datasets = np.empty(num_unique_worms, dtype=object)
        if i == 0:  # keep the validation dataset the same
            test_datasets = np.empty(num_unique_worms, dtype=object)
        neurons_masks = np.empty(num_unique_worms, dtype=object)
        # iterate over each worm in the cohort
        for j, (worm, single_worm_dataset) in enumerate(cohort):
            # check the memo for existing loaders and masks
            if worm in memo_loaders_masks:
                train_datasets[j] = memo_loaders_masks[worm]["train_dataset"]
                if i == 0:  # keep the validation dataset the same
                    test_datasets[j] = memo_loaders_masks[worm]["test_dataset"]
                train_mask = memo_loaders_masks[worm]["train_mask"]
                test_mask = memo_loaders_masks[worm]["test_mask"]
            # create data loaders and train/test masks only once per worm
            else:
                (
                    train_datasets[j],
                    tmp_test_dataset,
                    train_mask,
                    test_mask,
                ) = split_train_test(
                    data=single_worm_dataset[key_data],
                    time_vec=single_worm_dataset["time_in_seconds"],  # time vector
                    **kwargs,
                )
                if i == 0:  # keep the validation dataset the same
                    test_datasets[j] = tmp_test_dataset
                # add to memo
                memo_loaders_masks[worm] = dict(
                    train_dataset=train_datasets[j],
                    test_dataset=tmp_test_dataset,
                    train_mask=train_mask,
                    test_mask=test_mask,
                )
            # get the neurons mask for this worm
            neurons_masks[j] = single_worm_dataset["named_neurons_mask"]
            # update coverage mask
            coverage_mask |= single_worm_dataset["named_neurons_mask"]
            # mutate this worm's dataset with its train and test masks
            dataset[worm].setdefault("train_mask", train_mask.detach())
            dataset[worm].setdefault("test_mask", test_mask.detach())
            # increment the count of the amt. of train data (measure in timesteps) of one epoch
            if i == 0:
                worm_timesteps += train_mask.sum().item()
        # create full datasets and masks for the cohort
        neurons_mask = list(neurons_masks)
        train_dataset = ConcatDataset(list(train_datasets))
        test_dataset = ConcatDataset(list(test_datasets))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_sequences,  # shuffle sampled seqeuences
            pin_memory=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle_sequences,  # shuffle sampled seqeuences
            pin_memory=True,
            num_workers=0,
        )
        # optimize for 1 epoch per cohort
        num_epochs = 1  # 1 cohort = 1 epoch
        # get the starting timestamp
        start_time = time.perf_counter()
        # `optimize_model` can accepts the train and test data loaders and the neuron masks
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
        # get the ending timestamp
        end_time = time.perf_counter()
        # calculate the elapsed and average time
        n = i + 1
        seconds = end_time - start_time
        cumsum_seconds += seconds
        seconds_per_epoch = (n - 1) / n * seconds_per_epoch + 1 / n * seconds
        # retrieve losses and sample counts
        for key in data:  # pre-allocated memory for `data[key]`
            # with `num_epochs=1`, this is just data[key][i] = log[key]
            data[key][(i * num_epochs) : (i * num_epochs) + len(log[key])] = log[key]
        # extract the latest validation loss
        val_loss = data["centered_test_losses"][-1]
        # get what neurons have been covered so far
        covered_neurons = set(
            np.array(NEURONS_302)[torch.nonzero(coverage_mask).squeeze().numpy()]
        )
        num_covered_neurons = len(covered_neurons)
        # saving model checkpoints
        if (i % config.train.save_freq == 0) or (i + 1 == train_epochs):
            # display progress
            print(
                "Saving a model checkpoint.\n\tnum. worm cohorts trained on:",
                i,
                end="\n\n",
            )
            # save model checkpoints
            chkpt_name = "{}_epochs_{}_worms.pt".format(
                reset_epoch, i * num_unique_worms
            )
            checkpoint_path = os.path.join(log_dir, "checkpoints", chkpt_name)
            torch.save(
                {
                    # state dictionaries
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    # names
                    "dataset_name": dataset_name,
                    "model_name": model_class_name,
                    "optimizer_name": optim_name,
                    # training params
                    "epoch": reset_epoch,
                    "seq_len": config.train.seq_len,
                    "tau": config.train.tau_in,
                    "loss": val_loss,
                    "learning_rate": learn_rate,
                    "smooth_data": smooth_data,
                    # model instance params
                    "input_size": model.get_input_size(),
                    "hidden_size": model.get_hidden_size(),
                    "num_layers": model.get_num_layers(),
                    "loss_name": model.get_loss_name(),
                    "reg_param": model.get_reg_param(),
                    # other variables
                    "timestamp": timestamp,
                    "elapsed_time_seconds": cumsum_seconds,
                    "covered_neurons": covered_neurons,
                    "worm_timesteps": worm_timesteps,
                    "num_worm_cohorts": i,
                    "num_unique_worms": num_unique_worms,
                },
                checkpoint_path,
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
    # modify the config file
    config = OmegaConf.structured(OmegaConf.to_yaml(config))
    config.setdefault("dataset", {"name": dataset_name})
    config.dataset.name = dataset_name
    config.setdefault("model", {"type": model_class_name})
    config.setdefault(
        "predict",
        {"model": {"checkpoint_path": checkpoint_path.split("worm-graph/")[-1]}},
    )
    config.predict.model.checkpoint_path = checkpoint_path.split("worm-graph/")[-1]
    config.setdefault("visualize", {"log_dir": log_dir.split("worm-graph/")[-1]})
    config.visualize.log_dir = log_dir.split("worm-graph/")[-1]
    # add some global variables
    config.setdefault("globals", {"use_residual": False, "shuffle": False})
    config.globals.timestamp = timestamp
    config.globals.num_unique_worms = num_unique_worms
    config.globals.num_covered_neurons = num_covered_neurons
    config.globals.worm_timesteps = worm_timesteps
    config.globals.seconds_per_epoch = seconds_per_epoch
    # save config to log directory
    OmegaConf.save(config, os.path.join(log_dir, "config.yaml"))
    # delete the original (not updated) hydra config file
    og_cfg_file = os.path.join(log_dir, ".hydra", "config.yaml")
    if os.path.exists(og_cfg_file):
        os.remove(og_cfg_file)
    # returned trained model and a path to the log directory
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
