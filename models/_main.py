from models._utils import *

# Init logger
logger = logging.getLogger(__name__)

def get_model(model_config: DictConfig) -> torch.nn.Module:
    """Instantiate or load a model as specified in 'model.yaml'.

    This function can either create a new model of the specified type
    and size or load a saved model from a checkpoint file. The configuration
    is provided as a DictConfig object, which should include information
    about the model type, size, and other relevant parameters.

    Parameters
    ----------
    config : DictConfig
        A Hydra configuration object.

    Calls
    -----
    NeuralTransformer : class in models/_utils.py
    NetworkLSTM : class in models/_utils.py
    NeuralCFC : class in models/_utils.py
    LinearNN : class in models/_utils.py
        If no model type is specified, LinearNN is used by default.

    Returns
    -------
    model : torch.nn.Module
        A new or loaded model instance, moved to the appropriate device and
        cast to torch.float32 data type.

    """

    # If a checkpoint is given (True), load a saved model
    if model_config.use_this_pretrained_model:
        PATH = os.path.join(ROOT_DIR, model_config.use_this_pretrained_model)
        checkpoint = torch.load(PATH, map_location=torch.device(DEVICE))
        model_name = checkpoint["model_name"]
        input_size = checkpoint["input_size"]
        hidden_size = checkpoint["hidden_size"]
        num_layers = checkpoint["num_layers"]
        loss_name = checkpoint["loss_name"]
        fft_reg_param = checkpoint["fft_reg_param"]
        l1_reg_param = checkpoint["l1_reg_param"]
        model_state_dict = checkpoint["model_state_dict"]
        model = eval(model_name)(
            input_size,
            hidden_size,
            num_layers,
            loss=loss_name,
            fft_reg_param=fft_reg_param,
            l1_reg_param=l1_reg_param,
        )
        model.load_state_dict(model_state_dict)
        logger.info("Loading model from checkpoint: {}".format(model_config.use_this_pretrained_model))

    # Otherwise, instantiate a new model
    else:
        assert "type" in model_config, ValueError(
            "No model type or checkpoint path specified."
        )
        args = dict(
            input_size=model_config.input_size,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers,
            loss=model_config.loss,
            fft_reg_param=model_config.fft_reg_param,
            l1_reg_param=model_config.l1_reg_param,
        )
        if model_config.type == "NeuralTransformer":
            model = NeuralTransformer(**args)
        elif model_config.type == "NetworkLSTM":
            model = NetworkLSTM(**args)
        elif model_config.type == "NetworkRNN":
            model = NetworkRNN(**args)
        elif model_config.type == "NeuralCFC":
            model = NeuralCFC(**args)
        elif model_config.type == "NetworkGCN":
            model = NetworkGCN(**args)
        elif model_config.type == "LinearNN":
            model = LinearNN(**args)
        else:  # default to "LinearNN" model
            model = LinearNN(**args)
        logger.info("Initialized a new model: {}.".format(model_config.type))

    return model.to(torch.float32)


if __name__ == "__main__":
    config = OmegaConf.load("configs/submodule/model.yaml")
    print(OmegaConf.to_yaml(config), end="\n\n")
    model = get_model(config.model)
