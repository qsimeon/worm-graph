from preprocess._utils import *


def process_data(preprocess_config: DictConfig) -> None:
    """Preprocesses the raw neural and connectome data.

    This function preprocesses raw neural and connectome data to be used
    in downstream modeling and analysis tasks. It checks if the neural
    data and connectome data have been processed already; if not, it calls
    the appropriate functions to process and save them in the specified format.

    Params
    ------
    config: DictConfig
        Hydra configuration object. See configs submodule for details.

    Calls
    -----
    pickle_neural_data : function in preprocess/_utils.py
        Preprocess and convert neural data to .pickle format.
    preprocess_connectome : function in preprocess/_utils.py

    """

    # Init logger
    logger = logging.getLogger(__name__)

    # Pickle (and download) the data neural data if not already done
    if not os.path.exists(os.path.join(ROOT_DIR, "data/processed/neural/.processed")):
        logger.info("Preprocessing C. elegans neural data...")
        kwargs = dict(
            alpha=preprocess_config.smooth.alpha,
            window_size=preprocess_config.smooth.window_size,
            sigma=preprocess_config.smooth.sigma,
        )
        pickle_neural_data(
            url=preprocess_config.url,
            zipfile=preprocess_config.zipfile,
            dataset=preprocess_config.dataset,
            smooth_method=preprocess_config.smooth.method,
            resample_dt=preprocess_config.resample_dt,
            interpolate_method=preprocess_config.interpolate,
            cleanup=preprocess_config.cleanup,
            **kwargs,
        )
        logger.info("Finished preprocessing neural data.")
    else:
        logger.info("Neural data already preprocessed.")

    # Preprocess the connectome data if not already done
    if not os.path.exists(os.path.join(ROOT_DIR, "data/processed/connectome/graph_tensors.pt")):
        logger.info("Preprocessing C. elegans connectome...")
        raw_dir = os.path.join(ROOT_DIR, "data/raw")
        preprocess_connectome(raw_dir=raw_dir, raw_files=RAW_FILES)
        logger.info("Finished preprocessing C. elegans connectome.")
    else:
        logger.info("Connectome already preprocessed.")

    return None


if __name__ == "__main__":
    config = OmegaConf.load("configs/submodule/preprocess.yaml")
    print(OmegaConf.to_yaml(config), end="\n\n")
    process_data(config.preprocess)
