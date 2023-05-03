from models._utils import *


def get_model(config: DictConfig) -> torch.nn.Module:
    """
    Returns a new model of the type and size specified in 'model.yaml'.
    """

    # load a saved model
    if config.model.checkpoint_path:
        PATH = os.path.join(ROOT_DIR, config.model.checkpoint_path)
        checkpoint = torch.load(PATH, map_location=torch.device(DEVICE))
        model_name = checkpoint["model_name"]
        input_size = checkpoint["input_size"]
        hidden_size = checkpoint["hidden_size"]
        num_layers = checkpoint["num_layers"]
        loss_name = checkpoint["loss_name"]
        reg_param = checkpoint["reg_param"]
        model_state_dict = checkpoint["model_state_dict"]
        model = eval(model_name)(
            input_size,
            hidden_size,
            num_layers,
            loss=loss_name,
            reg_param=reg_param,
        )
        model.load_state_dict(model_state_dict)
        print("Model checkpoint path:", PATH, end="\n\n")
    # create a new model
    else:
        assert "type" in config.model, ValueError(
            "No model type or checkpoint path specified."
        )
        args = dict(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            loss=config.model.loss,
            reg_param=config.model.reg_param,
        )
        if config.model.type == "NeuralTransformer":
            model = NeuralTransformer(**args)
        elif config.model.type == "NetworkLSTM":
            model = NetworkLSTM(**args)
        elif config.model.type == "NeuralCFC":
            model = NeuralCFC(**args)
        elif config.model.type == "LinearNN":
            model = LinearNN(**args)
        else:  # default to "linear" model
            model = LinearNN(**args)
        print("Initialized a new model.", end="\n\n")
    print("Model:", model, end="\n\n")
    return model.to(torch.float32)


if __name__ == "__main__":
    config = OmegaConf.load("conf/model.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    get_model(config)
