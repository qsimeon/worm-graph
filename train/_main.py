from train._utils import *

# Init logger
logger = logging.getLogger(__name__)


def train_model(
    train_config: DictConfig,
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    verbose: bool = False,
):
    """
    Main standard ML training loop.

    Parameters
    ----------
    train_config : DictConfig
        A dictionary containing the training configuration.
    model : torch.nn.Module
        A model instance.
    train_dataset : torch.utils.data.Dataset
        A train dataset instance.
    val_dataset : torch.utils.data.Dataset
        A validation dataset instance.
    verbose : bool, optional
        Whether to print statistics at each epoch, by default False (True when using MULTIRUN hydra mode)

    Returns
    -------
    model: torch.nn.Module
        The trained model.
    metric: float
        The metric used for optuna (lowest validation loss across epochs).
    """
    # Verifications
    assert (
        isinstance(train_config.epochs, int) and train_config.epochs > 0
    ), "epochs must be a positive integer"
    assert (
        isinstance(train_config.batch_size, int) and train_config.batch_size > 0
    ), "batch_size must be a positive integer"
    assert isinstance(train_config.shuffle, bool), "shuffle must be a boolean"
    assert train_config.optimizer in [
        "SGD",
        "Adam",
        "AdamW",
        "Adagrad",
        "Adadelta",
        "RMSprop",
    ], "optimizer must be one of SGD, Adam, AdamW, Adagrad, Adadelta, RMSprop"
    # Initialize log directory
    log_dir = os.getcwd()  # logs/hydra/${now:%Y_%m_%d_%H_%M_%S}
    # Create train and checkpoints directories
    os.makedirs(os.path.join(log_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "train", "checkpoints"), exist_ok=True)
    # Extract hyperparams defined in the config file
    batch_size = train_config.batch_size
    shuffle = train_config.shuffle
    save_freq = train_config.save_freq
    optim_name = "torch.optim." + train_config.optimizer
    lr = 10 * train_config.lr if model.version_2 else train_config.lr  # starting learning rate
    # Loss function and number of epochs
    criterion = model.loss_fn()
    epochs = (
        train_config.epochs // 2 if model.version_2 else train_config.epochs
    )  # use fewer epochs for version 2
    # Instantiate early stopping
    early_stopper = EarlyStopping(
        patience=(
            train_config.early_stopping.patience // 2
            if model.version_2
            else train_config.early_stopping.patience
        ),
        min_delta=train_config.early_stopping.delta,
    )
    # Loss metrics to be stored
    # Training
    train_running_base_loss = 0
    train_running_loss = 0
    train_epoch_loss = []
    train_epoch_baseline = []
    # Validation
    val_running_base_loss = 0
    val_running_loss = 0
    val_epoch_loss = []
    val_epoch_baseline = []
    # Computation metrics
    learning_rate = []
    computation_time = []
    computation_flops = 0
    # Load model to device
    rank = DEVICE
    model = model.to(rank)
    # Initialize optimizer, learning rate scheduler and gradient scaler
    optimizer = eval(optim_name + "(model.parameters(), lr=" + str(lr) + ")")
    # TODO: Experiment with different learning rate schedulers.
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    # scheduler = lr_scheduler.CyclicLR(base_lr=0.1 * lr, max_lr=10 * lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs ** (1 / 3)), gamma=0.9)
    scaler = GradScaler()
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # num_workers=1,  # DEBUG
        pin_memory=False,
        # pin_memory="cuda" in DEVICE.type,  # DEBUG
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # num_workers=1,  # DEBUG
        pin_memory=False,
        # pin_memory="cuda" in DEVICE.type,  # DEBUG
    )
    # Start training
    logger.info("Starting training loop ...")
    pbar = tqdm(
        range(epochs + 1),  # +1 because we don't step() the first epoch
        position=0,
        leave=True,  # position at top and remove when done
        dynamic_ncols=True,  # adjust width to terminal window size
    )
    # ### DEBUG ###
    # # NOTE: This throws `RuntimeError: Function 'AbsBackward0' returned nan values in its 0th output.`
    # # when L1 regularization is added, indicating that some gradients are not computed correctly there.
    # torch.autograd.set_detect_anomaly(True, check_nan=True) # for debugging
    # ### DEBUG ###
    # Iterate over epochs
    for epoch in pbar:
        # ============================ Train loop ============================
        model.train()
        # Measure the time for an epoch
        start_time = time.perf_counter()
        # Training
        for batch_idx, (X_train, Y_train, mask_train, _) in enumerate(train_loader):
            X_train = X_train.to(rank)
            Y_train = Y_train.to(rank)
            mask_train = mask_train.to(rank)
            # Baseline model/naive predictor: predict that the next time step is the same as the current one.
            if model.version_2:
                train_baseline = torch.tensor(0.0)
            else:
                Y_base = X_train  # neural activity `[batch_size, seq_len, input_size]`
                train_baseline = criterion(output=Y_base, target=Y_train, mask=mask_train)
            # Reset / zero-out gradients
            optimizer.zero_grad()
            # Forward pass. Models are sequence-to-sequence.
            Y_pred = model(X_train, mask_train)
            train_loss = criterion(output=Y_pred, target=Y_train, mask=mask_train)
            ### DEBUG ###
            if math.isnan(train_loss.item()):
                print(f"Train loss: {train_loss.item()} | Train baseline: {train_baseline.item()}")
            ### DEBUG ###
            # Backpropagation.
            if epoch > 0:  # skip first epoch to get tabula rasa loss
                # Check if the computed loss requires gradient (e.g. the NaivePredictor model does not)
                if train_loss.requires_grad:
                    # Backward pass
                    scaler.scale(train_loss).backward()
                    # Update model weights
                    scaler.step(optimizer)
                    # Update the grad scaler
                    scaler.update()
            # Calculate total FLOP only at the first epoch and batch
            elif batch_idx == 0:
                # TODO: Find way to compute FLOP using Pytorch Profiler
                computation_flops = 0  # currently unused and thus set to 0
            # Update running metrics
            train_running_base_loss += train_baseline.item()
            train_running_loss += train_loss.item()
        # Compute the time taken
        end_time = time.perf_counter()
        computation_time.append(end_time - start_time)
        # Store metrics
        train_epoch_baseline.append(train_running_base_loss / len(train_loader))
        train_epoch_loss.append(train_running_loss / len(train_loader))
        # Reset metrics
        train_running_base_loss = 0
        train_running_loss = 0
        # ============================ Validation loop ============================
        model.eval()
        with torch.no_grad():
            # Validation
            for batch_idx, (X_val, Y_val, mask_val, _) in enumerate(val_loader):
                X_val = X_val.to(rank)
                Y_val = Y_val.to(rank)
                mask_val = mask_val.to(rank)
                # Baseline model/naive predictor: predict that the next time step is the same as the current one.
                if model.version_2:
                    val_baseline = torch.tensor(0.0)
                else:
                    Y_base = X_val  # neural activity `[batch_size, seq_len, input_size]`
                    val_baseline = criterion(output=Y_base, target=Y_val, mask=mask_val)
                # Forward pass. Models are sequence-to-sequence.
                Y_pred = model(X_val, mask_val)
                val_loss = criterion(output=Y_pred, target=Y_val, mask=mask_val)
                # Update running losses
                val_running_base_loss += val_baseline.item()
                val_running_loss += val_loss.item()
            # Store metrics
            val_epoch_loss.append(val_running_loss / len(val_loader))
            val_epoch_baseline.append(val_running_base_loss / len(val_loader))
            # Reset running losses
            val_running_base_loss = 0
            val_running_loss = 0
        # Step the scheduler
        scheduler.step(val_epoch_loss[-1])
        # Store current learning rate
        learning_rate.append(optimizer.param_groups[0]["lr"])
        # Save model checkpoint
        if epoch % save_freq == 0:
            save_model_checkpoint(
                model,
                os.path.join(log_dir, "train", "checkpoints", "model_epoch_" + str(epoch) + ".pt"),
                other_info={
                    "computation_flops": computation_flops,
                    "time_last_epoch": computation_time[-1],
                    "current_lr": learning_rate[-1],
                },  # save additional info to checkpoint
            )
        # Early stopping
        if early_stopper(model, val_epoch_loss[-1]):  # on validation loss
            logger.info(
                "Early stopping triggered (epoch: {} \t loss: {}).".format(
                    epoch, val_epoch_loss[-1]
                )
            )
            break
        # Print training progress metrics if in verbose mode
        if verbose:
            logger.info(
                f"Epoch: {epoch}/{epochs} | Learning rate: {learning_rate[-1]:.1e} | "
                f"Train loss: {train_epoch_loss[-1]:.3f} | Train time (s): {computation_time[-1]:.3f} | "
                f"Val. loss: {val_epoch_loss[-1]:.3f} | Val. baseline: {val_epoch_baseline[-1]:.3f} | "
            )
        # Update progress bar
        pbar.set_description(f"Epoch {epoch}/{epochs}")
        pbar.set_postfix({"Train loss": train_epoch_loss[-1], "Val. loss": val_epoch_loss[-1]})
    # Restore best model and save it with additional info
    logger.info("Training loop is over. Loading best model.")
    model.load_state_dict(early_stopper.best_model.state_dict())
    save_model_checkpoint(
        model,
        os.path.join(log_dir, "train", "checkpoints", "model_best.pt"),
        other_info={
            "computation_flops": computation_flops,
            "time_last_epoch": computation_time[-1],
            "current_lr": learning_rate[-1],
        },  # add additional info to checkpoint
    )
    logger.info(
        f"FLOP: {computation_flops}, \t Time (s) last epoch: {computation_time[-1]}, \t Parameter counts (total, trainable): {print_parameters(model, verbose=False)}"
    )
    # Save training and evaluation metrics into a csv file
    train_metrics = pd.DataFrame(
        {
            "epoch": np.arange(len(train_epoch_loss)),
            "computation_time": computation_time,
            "learning_rate": learning_rate,
            "train_loss": train_epoch_loss,
            "train_baseline": train_epoch_baseline,
            "val_loss": val_epoch_loss,
            "val_baseline": val_epoch_baseline,
        }
    )
    train_metrics.to_csv(os.path.join(log_dir, "train", "train_metrics.csv"), index=False)
    # Metric for Optuna (lowest validation loss)
    metric = min(val_epoch_loss)
    if metric == np.nan or metric is None:
        metric = float("inf")
    return model, metric


if __name__ == "__main__":
    # Get the appropriate configs
    train_config = OmegaConf.load("configs/submodule/train.yaml")
    print(OmegaConf.to_yaml(train_config), end="\n\n")
    model_config = OmegaConf.load("configs/submodule/model.yaml")
    print(OmegaConf.to_yaml(model_config), end="\n\n")
    dataset_config = OmegaConf.load("configs/submodule/dataset.yaml")
    print(OmegaConf.to_yaml(dataset_config), end="\n\n")
    # Create new log directory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join("logs/hydra", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    # Switch to the log directory
    os.chdir(log_dir)
    # Dataset
    train_dataset, val_dataset = get_datasets(dataset_config.dataset)
    # Get the model
    model = get_model(model_config.model)
    # Train the model
    model, metric = train_model(
        train_config=train_config.train,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    print(f"Final metric: \t {metric}\n")
