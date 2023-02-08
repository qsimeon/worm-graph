from _utils import *


@hydra.main(version_base=None, config_path=".", config_name="model")
def get_model(config):
    """
    Use something search based for logging.
    Think about canconical plots that you always want to make:
        - e.g. bunch of curves where I hold all else constant except
            1 config item
        - each different config line a different color.
    Add unit tests for each config.
    """
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
