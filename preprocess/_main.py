from _utils import *


@hydra.main(version_base=None, config_path=".", config_name="preprocess")
def process_data(config):
    pickle_neural_data(
        url=config.url,
        zipfile=config.zipfile,
        dataset=config.dataset,
    )
    print("C. elegans neural data has been pickled!", end="\n\n")
    preprocess_connectome(raw_dir=config.raw_dir, raw_files=RAW_FILES)
    print("C. elegans connectome has been preprocessed!", end="\n\n")
    return None


if __name__ == "__main__":
    process_data()
