from models._utils import *

# Init logger
logger = logging.getLogger(__name__)


def get_model(model_config: DictConfig, verbose=True) -> torch.nn.Module:
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
    PureAttention : class in models/_utils.py
    NeuralTransformer : class in models/_utils.py
    NetworkLSTM : class in models/_utils.py
    NetworkCTRNN : class in models/_utils.py
    LiquidCfC : class in models/_utils.py
    FeatureFFNN : class in models/_utils.py
    NaivePredictor : class in models/_utils.py
    LinearRegression: class in models/_utils.py
        If no model type is specified, LinearRegression is used by default.

    Returns
    -------
    model : torch.nn.Module
        A new or loaded model instance, moved to the appropriate device and
        cast to torch.float32 data type.

    """

    # If a checkpoint is given (not None), load a saved model
    if model_config.use_this_pretrained_model:
        PATH = os.path.join(ROOT_DIR, model_config.use_this_pretrained_model)
        checkpoint = torch.load(PATH, map_location=DEVICE)
        model_name = checkpoint["model_name"]
        input_size = checkpoint["input_size"]
        hidden_size = checkpoint["hidden_size"]
        loss_name = checkpoint["loss_name"]
        l1_reg_param = checkpoint["l1_reg_param"]
        # version_2 = checkpoint["version_2"] # TODO
        model_state_dict = checkpoint["model_state_dict"]
        model = eval(model_name)(
            input_size,
            hidden_size,
            loss=loss_name,
            l1_reg_param=l1_reg_param,
        )
        model.load_state_dict(model_state_dict)
        if verbose:
            logger.info(
                "Loading model from checkpoint: {}".format(model_config.use_this_pretrained_model)
            )
            logger.info(f"Hidden size: {hidden_size}")

    # Otherwise, instantiate a new model
    else:
        assert "type" in model_config, ValueError("No model type or checkpoint path specified.")
        args = dict(
            input_size=model_config.input_size,
            hidden_size=model_config.hidden_size,
            loss=model_config.loss,
            l1_reg_param=model_config.l1_reg_param,
            # version_2=model_config.version_2, # TODO
        )
        if model_config.type == "PureAttention":
            model = PureAttention(**args)
        elif model_config.type == "NeuralTransformer":
            model = NeuralTransformer(**args)
        elif model_config.type == "NetworkLSTM":
            model = NetworkLSTM(**args)
        elif model_config.type == "NetworkCTRNN":
            model = NetworkCTRNN(**args)
        elif model_config.type == "LiquidCfC":
            model = LiquidCfC(**args)
        elif model_config.type == "FeatureFFNN":
            model = FeatureFFNN(**args)
        elif model_config.type == "NaivePredictor":
            model = NaivePredictor(**args)
        elif model_config.type == "LinearRegression":
            model = LinearRegression(**args)
        else:  # default to "LinearRegression" model
            model = LinearRegression(**args)

        if verbose:
            logger.info("Initialized a new model: {}.".format(model_config.type))
            logger.info(f"Hidden size: {model_config.hidden_size}")

    return model.to(torch.float32)


if __name__ == "__main__":
    config = OmegaConf.load("configs/submodule/model.yaml")
    print(OmegaConf.to_yaml(config), end="\n\n")
    model = get_model(config.model)
