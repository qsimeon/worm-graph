from preprocess._utils import *


def process_data(config: DictConfig) -> None:
    """Preprocesses the raw neural and connectome data.

    This function preprocesses raw neural and connectome data to be used
    in downstream modeling and analysis tasks. It checks if the neural
    data and connectome data have been processed already; if not, it calls
    the appropriate functions to process and save them in the specified format.

    Params
    ------
    config: DictConfig
        Hydra configuration object.
    
    Calls
    -----
    pickle_neural_data : function in preprocess/_utils.py
        Convert neural data to .pickle format.
    preprocess_connectome : function in preprocess/_utils.py

    Returns
    -------
    None
        The function's primary purpose is to preprocess the data 
        and save it to disk for future use.
    """

    # Pickle (and download) the data neural data if not already done
    if not os.path.exists(os.path.join(ROOT_DIR, "data/processed/neural/.processed")):
        pickle_neural_data(
            url=config.preprocess.url,
            zipfile=config.preprocess.zipfile,
            dataset=config.preprocess.dataset,
            smooth_method=config.preprocess.smooth,
            resample_dt=config.preprocess.resample_dt,
        )
        print("C. elegans neural data has been pickled!", end="\n\n")
    else:
        print("Neural data already pickled.", end="\n\n")

    # Preprocess the connectome data if not already done
    if not os.path.exists(
        os.path.join(ROOT_DIR, "data/processed/connectome/graph_tensors.pt")
    ):
        preprocess_connectome(raw_dir=config.preprocess.raw_dir, raw_files=RAW_FILES)
        print("C. elegans connectome has been preprocessed!", end="\n\n")
    else:
        print("Connectome already preprocessed.", end="\n\n")

    return None


if __name__ == "__main__":
    config = OmegaConf.load("conf/preprocess.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    process_data(config)
