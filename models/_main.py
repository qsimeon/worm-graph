from _utils import *


@hydra.main(version_base=None, config_path=".", config_name="model")
def get_model(config):
    # create the model
    if config.type == "lstm":
        model = NetworkLSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
        ).double()
    else:
        model = LinearNN(input_size=config.input_size).double()
    print(model)
    return model


if __name__ == "__main__":
    get_model()
