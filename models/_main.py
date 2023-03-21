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
    if config.model.type.lower() == "networklstm":
        model = NetworkLSTM(**args).double()
    elif config.model.type.lower() == "variationallstm":
        model = VariationalLSTM(**args).double()
    elif config.model.type.lower() == "neuralcfc":
        model = NeuralCFC(**args).double()
    elif config.model.type.lower() == "densecfc":
        model = DenseCFC(**args).double()
    elif config.model.type.lower() == "linearnn":
        model = LinearNN(**args).double()
    else:  # default to "linear" model
        model = LinearNN(**args).double()
    print("Model:", model, end="\n\n")
    return model


if __name__ == "__main__":
    config = OmegaConf.load("conf/model.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    get_model(config)
