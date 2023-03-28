from preprocess._utils import *


def process_data(config: DictConfig) -> None:
    """
    Preprocesses the raw neural and connectmoe data
    needed for downstream modeling and analysis.
    """
    if not os.path.exists(os.path.join(ROOT_DIR, "data/processed/neural/.processed")):
        pickle_neural_data(
            url=config.preprocess.url,
            zipfile=config.preprocess.zipfile,
            dataset=config.preprocess.dataset,
            smooth_method=config.preprocess.smooth,
        )
        print("C. elegans neural data has been pickled!", end="\n\n")
    else:
        print("Neural data already pickled.", end="\n\n")
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

    # # use this if you need to
    # data_loader = create_four_sine_datasets()
    # data_loader.main_create()
