from models._utils import *


def get_model(config: DictConfig) -> torch.nn.Module:
    """
    Returns a new model of the type and size specified in 'model.yaml'.
    """
    # create the model
    if config.model.type == "lstm":
        model = NetworkLSTM(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
        ).double()
    else:  # "linear" model
        model = LinearNN(input_size=config.model.input_size).double()
    print("Model:", model, end="\n\n")
    return model


if __name__ == "__main__":
    get_model(OmegaConf.load("conf/model.yaml"))
