from preprocess._utils import *


def process_data(config: DictConfig) -> None:
    """
    Preprocesses the raw neural and connectmoe data
    needed for downstream modeling and analysis.
    """
    if not os.path.exists("data/processed/neural/.processed"):
        pickle_neural_data(
            url=config.preprocess.url,
            zipfile=config.preprocess.zipfile,
            dataset=config.preprocess.dataset,
        )
        print("C. elegans neural data has been pickled!", end="\n\n")
    else:
        print("Neural data already pickled.", end="\n\n")
    if not os.path.exists("data/processed/connectome/graph_tensors.pt"):
        preprocess_connectome(raw_dir=config.preprocess.raw_dir, raw_files=RAW_FILES)
        print("C. elegans connectome has been preprocessed!", end="\n\n")
    else:
        print("Connectome already preprocessed.", end="\n\n")
    return None


if __name__ == "__main__":
    process_data(OmegaConf.load("conf/preprocess.yaml"))
