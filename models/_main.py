from models._utils import *


def get_model(config: DictConfig) -> torch.nn.Module:
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

    Notes
    -----
    """

    # If a checkpoint is given (True), load a saved model
    if config.model.checkpoint_path:
        PATH = os.path.join(ROOT_DIR, config.model.checkpoint_path)
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
        print("Model checkpoint path:", PATH, end="\n\n")

    # Otherwise, instantiate a new model
    else:
        assert "type" in config.model, ValueError(
            "No model type or checkpoint path specified."
        )
        args = dict(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            loss=config.model.loss,
            fft_reg_param=config.model.fft_reg_param,
            l1_reg_param=config.model.l1_reg_param,
        )
        if config.model.type == "NeuralTransformer":
            model = NeuralTransformer(**args)
        elif config.model.type == "NetworkLSTM":
            model = NetworkLSTM(**args)
        elif config.model.type == "NetworkRNN":
            model = NetworkRNN(**args)
        elif config.model.type == "NeuralCFC":
            model = NeuralCFC(**args)
        elif config.model.type == "NetworkGCN":
            model = NetworkGCN(**args)
        elif config.model.type == "LinearNN":
            model = LinearNN(**args)
        else:  # default to "LinearNN" model
            model = LinearNN(**args)
        print("Initialized a new model.", end="\n\n")

    print("Model:", model, end="\n\n")

    return model.to(torch.float32)


if __name__ == "__main__":
    config = OmegaConf.load("conf/model.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    model = get_model(config)
