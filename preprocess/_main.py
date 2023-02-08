from _utils import *


@hydra.main(version_base=None, config_path=".", config_name="preprocess")
def process_data(config):
    pickle_neural_data(
        url=config.url,
        zipfile=config.zipfile,
        dataset=config.dataset,
    )
    return None


if __name__ == "__main__":
    process_data()
