from models._utils import *


def get_model(config: DictConfig) -> torch.nn.Module:
    """
    Returns a new model of the type and size specified in 'model.yaml'.
    """
    # create the model
    args = dict(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_layers=1,
        loss=config.model.loss,
    )
    if config.model.type == "NetworkLSTM":
        model = NetworkLSTM(**args)
    elif config.model.type == "NeuralCFC":
        model = NeuralCFC(**args)
    elif config.model.type == "LinearNN":
        model = LinearNN(**args)
    else:  # default to "linear" model
        model = LinearNN(**args)
    print("Model:", model, end="\n\n")
    return model.to(torch.float32)


if __name__ == "__main__":
    config = OmegaConf.load("conf/model.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    get_model(config)
