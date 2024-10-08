from train._pkg import *

# Init logger
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs. Code adapted from:
    https://github.com/jeffheaton/t81_558_deep_learning/blob/pytorch/t81_558_class_03_4_early_stop.ipynb
    """

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_model = None
        self.best_loss = None
        self.counter = 0

    def __call__(self, model, val_loss):
        if val_loss is None or math.isnan(val_loss) or math.isinf(val_loss):
            if self.best_model is None:
                self.best_model = model
            if self.best_loss is None:
                self.best_loss = float("inf")
                self.best_model = model
            ## DEBUG ###
            # Return early immediately if the validation loss is not a valid number?
            logger.info(f"Stopping early. Validation loss {val_loss} is not a valid number.")
            return True
            ## DEBUG ####
            # logger.info(f"Validation loss {val_loss} is not a valid number.")
            # return False
        if self.best_model is None:
            self.best_model = model
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
        if self.counter >= self.patience:
            logger.info(f"Stopping early. Loss has not improved in {self.patience} epochs.")
            return True
        return False


def save_model_checkpoint(model, checkpoint_path, other_info=dict()):
    """
    Saves a PyTorch model to disk. Saves everything needed to load the
    model again using the function `model._utils.load_model_checkpoint`.

    Args:
        model (nn.Module): The PyTorch model to save.
        checkpoint_path (str): The path to save the model to.
        other_info (dict, optional): Any additional information to save with the model. Defaults to an empty dictionary.
    """
    # num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # DEBUG
    _, num_trainable_params = print_parameters(model, verbose=False)
    base_info = {
        # State dictionaries
        "model_state_dict": model.state_dict(),
        # Names
        "model_name": model.__class__.__name__,
        # Number of parameters
        "num_trainable_params": num_trainable_params,
        # Model parameters
        "input_size": model.get_input_size(),
        "hidden_size": model.get_hidden_size(),
        "loss_name": model.get_loss_name(),
        "l1_norm_reg_param": model.get_l1_norm_reg_param(),
        "connectome_reg_param": model.get_connectome_reg_param(),
        # New attributes for version 2
        "version_2": model.version_2,
        "num_tokens": model.num_tokens if model.version_2 else None,
    }
    save_info = {**base_info, **other_info}
    torch.save(save_info, checkpoint_path)


############################## Constructing the process group for distributed training

# Rank is the id of each process. In this case, for each GPU. Indexing starts by zero.
# World_size is the total number of processes. In this case, the number of GPUs.


def ddp_setup(rank, world_size=WORLD_SIZE):
    """
    Set up distributed data training.

    Args:
        rank (int): Unique identifier for each gpu
        world_size (int): Total number of processes
    """
    init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )  # nccl für cuda, gloo für cpu
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True


##############################
