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
    elif config.model.type == "vae_lstm":
        model = VAELSTM(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            num_layers=1,
        ).double()
    elif config.model.type == "neural_cfc":
        model = NeuralCFC(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            num_layers=1,
        ).double()
    else:  # "linear" model
        model = LinearNN(
            input_size=config.model.input_size, 
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
        ).double()
    print("Model:", model, end="\n\n")
    return model


if __name__ == "__main__":
    config = OmegaConf.load("conf/model.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    get_model(config)
