from model._utils import *

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
    PureAttention : class in model/_utils.py
    NeuralTransformer : class in model/_utils.py
    NetworkLSTM : class in model/_utils.py
    HippoSSM : class in model/_utils.py
    NetworkCTRNN : class in model/_utils.py
    LiquidCfC : class in model/_utils.py
    FeatureFFNN : class in model/_utils.py
    NaivePredictor : class in model/_utils.py
    LinearRegression: class in model/_utils.py
        If no model type is specified, LinearRegression is used by default.

    Returns
    -------
    model : torch.nn.Module
        A new or loaded model instance, moved to the appropriate device and
        cast to torch.float32 data type.

    """

    # If a checkpoint is given (not None), load a saved model
    if model_config.use_this_pretrained_model:
        # Load the model from the checkpoint path
        checkpoint_path = os.path.join(ROOT_DIR, model_config.use_this_pretrained_model)
        model = load_model_checkpoint(checkpoint_path)
        # Some logging of checkpoint and model information
        if verbose:
            logger.info(
                "Loaded model from checkpoint: {}".format(model_config.use_this_pretrained_model)
            )
            logger.info(f"Hidden size: {model.get_hidden_size()}")

    # Otherwise, instantiate a new model
    else:
        assert "type" in model_config, ValueError("No model type or checkpoint path specified.")
        ### DEBUG ###
        # If set to null in the model config, we default to the glonbal variable `NUM_NEURONS`
        if model_config.input_size is None:
            model_config.input_size = NUM_NEURONS
        ### DEBUG ###
        args = dict(
            input_size=model_config.input_size,
            hidden_size=model_config.hidden_size,
            loss=model_config.loss,
            l1_norm_reg_param=model_config.l1_norm_reg_param,
            connectome_reg_param=model_config.connectome_reg_param,
        )
        if model_config.type == "PureAttention":
            model = PureAttention(**args)
        elif model_config.type == "NeuralTransformer":
            model = NeuralTransformer(**args)
        elif model_config.type == "NetworkLSTM":
            model = NetworkLSTM(**args)
        elif model_config.type == "HippoSSM":
            model = HippoSSM(**args)
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

    return model


if __name__ == "__main__":
    config = OmegaConf.load("configs/submodule/model.yaml")
    print(OmegaConf.to_yaml(config), end="\n\n")
    model = get_model(config.model)
    print(f"Got model : \n\t {model} \n")
