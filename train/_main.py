from train._utils import *

# Init logger
logger = logging.getLogger(__name__)

def train_model(
    train_config: DictConfig,
    model: torch.nn.Module,
    dataset: dict,
) -> tuple[torch.nn.Module, str]:
    """Trains a neural network model on a multi-worm dataset.

    The function saves the training progress, loss curves, and model
    checkpoints during training. This function takes in a configuration
    dictionary, model, dataset, and optional parameters to control the
    training process. It returns the trained model and the path to the
    directory containing the training and evaluation logs.

    Parameters
    ----------
    config : DictConfig
        Hydra configuration object containing training parameters.
    model : torch.nn.Module
        The neural network model to be trained.
    dataset : dict
        A dictionary containing the multi-worm dataset.
    shuffle_worms : bool, optional
        Whether to shuffle all worms. Defaults to True.
    log_dir : str or None, optional
        Path to the directory where training and evaluation logs will be
        saved. If None, the current working directory will be used.
        Defaults to None.

    Calls
    -----
    split_train_test : function in train/_utils.py
        Splits data into train and test sets for a single worm.
    optimize_model : function in train/_utils.py
        Trains and validates the model for the specified number of epochs.

    Returns
    -------
    tuple : (torch.nn.Module, str)
        A tuple containing the trained model and the path to the directory
        with training and evaluation logs.

    Notes
    -----
    * You can find the loss curves and checkpoints of the trained model in
      the `log_dir` directory.
    * Calcium data is used by default. Set use_residual to True in main.yaml
      to use the residual data instead.
    * No smoothening is applied to the data by default. Set smooth_data to
      True in main.yaml to use the smoothened data instead.
    """

    # Check if we have a valid dataset (must have at least one worm: "worm0")
    assert "worm0" in dataset, "Not a valid dataset object."

    # Get some helpful variables
    dataset_name = dataset["worm0"]["dataset"]
    model_class_name = model.__class__.__name__
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    num_unique_worms = len(dataset)
    train_epochs = train_config.epochs + 1
    shuffle_samples = train_config.shuffle_samples
    batch_size = train_config.num_samples // train_config.num_batches
    use_residual = train_config.use_residual
    smooth_data = train_config.use_smooth_data
    shuffle_worms = train_config.shuffle_worms

    log_dir = os.getcwd()  # logs/hydra/${now:%Y_%m_%d_%H_%M_%S}

    # Create a model checkpoints folder
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    # Cycle the dataset until the desired number epochs obtained
    dataset_items = sorted(dataset.items()) * train_epochs
    assert (
        len(dataset_items) == train_epochs * num_unique_worms
    ), "Invalid number of worms."

    # Shuffle (without replacement) the worms (including duplicates) in the dataset
    if shuffle_worms:
        dataset_items = random.sample(dataset_items, k=len(dataset_items))

    # Split the dataset into cohorts of worms; there should be one cohort per epoch
    worm_cohorts = np.array_split(dataset_items, train_epochs)
    worm_cohorts = [_.tolist() for _ in worm_cohorts]  # Convert cohort arrays to lists
    assert len(worm_cohorts) == train_epochs, "Invalid number of cohorts."
    assert all(
        [len(cohort) == num_unique_worms for cohort in worm_cohorts]
    ), "Invalid cohort size."

    # Move model to device
    model = model.to(DEVICE)

    # Instantiate the optimizer
    opt_param = train_config.optimizer
    optim_name = "torch.optim." + opt_param
    learn_rate = train_config.learn_rate

    if train_config.optimizer is not None:
        if isinstance(opt_param, str):
            optimizer = eval(
                optim_name + "(model.parameters(), lr=" + str(learn_rate) + ")"
            )
        assert isinstance(
            optimizer, torch.optim.Optimizer
        ), "Please use an instance of torch.optim.Optimizer."
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

    # Early stopping
    es = EarlyStopping(
        patience = train_config.early_stopping.patience,
        min_delta = train_config.early_stopping.delta,
        restore_best_weights = train_config.early_stopping.restore_best_weights,
        )

    # Initialize train/test loss metrics arrays
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

    # Create keyword arguments for `split_train_test`
    kwargs = dict(
        k_splits=train_config.k_splits,
        seq_len=train_config.seq_len,
        num_samples=train_config.num_samples,  # Number of samples per worm
        reverse=train_config.reverse,  # Generate samples backward from the end of data
        tau=train_config.tau_in,
        use_residual=use_residual,
    )

    # Choose whether to use calcium or residual data
    if use_residual:
        key_data = "residual_calcium"
    else:
        key_data = "calcium_data"

    # Choose whether to use original or smoothed data
    if smooth_data:
        key_data = "smooth_" + key_data
    else:
        key_data = key_data

    # Throughout training, keep track of what neurons have been covered
    coverage_mask = torch.zeros_like(
        dataset["worm0"]["named_neurons_mask"], dtype=torch.bool
    )
    # Memoize creation of data loaders and masks for speedup
    memo_loaders_masks = dict()
    # Train for train_config.num_epochs
    reset_epoch = 0
    # Keep track of the amount of train data per epoch (in timesteps)
    worm_timesteps = 0
    # Keep track of the total optimization time
    cumsum_seconds = 0
    # Keep track of the average optimization time per epoch
    seconds_per_epoch = 0

    logger.info(f"Start training.")

    # Main FOR loop; train the model for multiple epochs (one cohort = one epoch)
    # In one epoch we process all worms (i.e one cohort)
    pbar = tqdm(
        enumerate(worm_cohorts), 
        total=len(worm_cohorts),
        position=0, leave=True,  # position at top and remove when done
        dynamic_ncols=True,  # adjust width to terminal window size
        )
    
    for i, cohort in pbar:

        # Create a array of datasets and masks for the cohort
        train_datasets = np.empty(num_unique_worms, dtype=object)

        if i == 0:  # Keep the validation dataset the same
            test_datasets = np.empty(num_unique_worms, dtype=object)

        neurons_masks = np.empty(num_unique_worms, dtype=object)

        # Iterate over each worm in the cohort
        for j, (worm, single_worm_dataset) in enumerate(cohort):
            # Check the memo for existing loaders and masks
            if worm in memo_loaders_masks:
                train_datasets[j] = memo_loaders_masks[worm]["train_dataset"]
                if i == 0:  # Keep the validation dataset the same
                    test_datasets[j] = memo_loaders_masks[worm]["test_dataset"]
                train_mask = memo_loaders_masks[worm]["train_mask"]
                test_mask = memo_loaders_masks[worm]["test_mask"]

            # Create data loaders and train/test masks only once per worm
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
                if i == 0:  # Keep the validation dataset the same
                    test_datasets[j] = tmp_test_dataset
                    
                # Add to memo
                memo_loaders_masks[worm] = dict(
                    train_dataset=train_datasets[j],
                    test_dataset=tmp_test_dataset,
                    train_mask=train_mask,
                    test_mask=test_mask,
                )

            # Get the neurons mask for this worm
            neurons_masks[j] = single_worm_dataset["named_neurons_mask"]
            # Update coverage mask
            coverage_mask |= single_worm_dataset["named_neurons_mask"]

            # Mutate this worm's dataset with its train and test masks
            dataset[worm].setdefault("train_mask", train_mask.detach())
            dataset[worm].setdefault("test_mask", test_mask.detach())

            # Increment the count of the amount of train data (measured in timesteps) of one epoch
            if i == 0:
                worm_timesteps += train_mask.sum().item()

        # Create full datasets and masks for the cohort
        neurons_mask = list(neurons_masks)
        train_dataset = ConcatDataset(list(train_datasets))
        test_dataset = ConcatDataset(list(test_datasets))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_samples,  # shuffle sampled sequences
            pin_memory=True,
            num_workers=0,
        )  # (X, Y, Dict)

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle_samples,  # shuffle sampled sequences
            pin_memory=True,
            num_workers=0,
        )  # (X, Y, Dict)

        # Optimize for 1 epoch per cohort
        num_epochs = 1  # 1 cohort = 1 epoch

        # Get the starting timestamp
        start_time = time.perf_counter()

        # `optimize_model` accepts the train and test data loaders, the neuron masks and optimizes for num_epochs
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

        # Get the ending timestamp
        end_time = time.perf_counter()

        # Calculate the elapsed and average time
        n = i + 1
        seconds = end_time - start_time
        cumsum_seconds += seconds
        seconds_per_epoch = (
            n - 1
        ) / n * seconds_per_epoch + 1 / n * seconds  # running average

        # Retrieve losses and sample counts
        for key in data:  # pre-allocated memory for `data[key]`
            # with `num_epochs=1`, this is just data[key][i] = log[key]
            data[key][(i * num_epochs) : (i * num_epochs) + len(log[key])] = log[key]

        # Extract the latest validation loss
        val_loss = data["centered_test_losses"][-1]

        # Get what neurons have been covered (i.e. seen) so far
        covered_neurons = set(
            np.array(NEURONS_302)[torch.nonzero(coverage_mask).squeeze().numpy()]
        )
        num_covered_neurons = len(covered_neurons)

        # Saving model checkpoints
        if (i % train_config.save_freq == 0) or (i + 1 == train_epochs):
            # logger.info("Saving a model checkpoint ({} epochs).".format(i))

            # Save model checkpoints
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
                    "seq_len": train_config.seq_len,
                    "tau": train_config.tau_in,
                    "loss": val_loss,
                    "learning_rate": learn_rate,
                    "smooth_data": smooth_data,
                    # model instance params
                    "input_size": model.get_input_size(),
                    "hidden_size": model.get_hidden_size(),
                    "num_layers": model.get_num_layers(),
                    "loss_name": model.get_loss_name(),
                    "fft_reg_param": model.get_fft_reg_param(),
                    "l1_reg_param": model.get_l1_reg_param(),
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

        if es(model, log['test_losses'][0]):
            logger.info("Early stopping triggered (epoch {}).".format(i))
            break
        
        # Update progress bar
        tqdm_description = f'Epoch {i+1}/{len(worm_cohorts)}'
        tqdm_postfix = {'train_loss': log['train_losses'][0], 'val_loss': log['test_losses'][0]}  # assuming train_loss is defined
        pbar.set_description(tqdm_description)
        pbar.set_postfix(tqdm_postfix)

        # Set to next epoch before continuing
        reset_epoch = log["epochs"][-1] + 1
        continue

    # Save loss curves
    pd.DataFrame(data=data).to_csv(
        os.path.join(log_dir, "loss_curves.csv"),
        index=True,
        header=True,
    )

    # Configs to update
    submodules_updated = OmegaConf.create({
        'dataset': {'train': {}},
        'model': {},
        'visualize': {},
        }
    )
    OmegaConf.update(submodules_updated.dataset.train, "name", dataset_name, merge=True) # updated dataset name
    OmegaConf.update(submodules_updated.model, "checkpoint_path", checkpoint_path.split("worm-graph/")[-1], merge=True) # updated checkpoint path
    OmegaConf.update(submodules_updated.visualize, "log_dir", log_dir, merge=True) # update visualize directory

    # Save train info
    train_info = OmegaConf.create({
        'timestamp': timestamp,
        'num_unique_worms': num_unique_worms,
        'num_covered_neurons': num_covered_neurons,
        'worm_timesteps': worm_timesteps,
        'seconds_per_epoch': seconds_per_epoch,
        }
    )

    # returned trained model, an update to the submodules and the train info
    return model, submodules_updated, train_info


if __name__ == "__main__":
    train_config = OmegaConf.load("configs/submodule/train.yaml")
    print(OmegaConf.to_yaml(train_config), end="\n\n")

    model_config = OmegaConf.load("configs/submodule/model.yaml")
    print(OmegaConf.to_yaml(model_config), end="\n\n")

    dataset_config = OmegaConf.load("configs/submodule/dataset.yaml")
    print(OmegaConf.to_yaml(dataset_config), end="\n\n")

    # Create new to log directory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join("logs/hydra", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # Move to new log directory
    os.chdir(log_dir)

    # Dataset
    train_dataset = get_dataset(dataset_config.dataset.train)

    # Get the model
    model = get_model(model_config.model)

    # Train the model
    model, submodules_updated, train_info = train_model(
        train_config=train_config.train,
        model=model,
        dataset=train_dataset,
    )

