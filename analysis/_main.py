from analysis._utils import *

# Init logger
logger = logging.getLogger(__name__)


def analyse_run(analysis_config: DictConfig):
    log_dir = analysis_config.analyse_this_log_dir

    assert log_dir is not None, "log_dir is None. Please specify a log directory to analyse."

    # Analyse validation loss across datasets
    loss_per_dataset(
        log_dir=log_dir,
        experimental_datasets=analysis_config.validation.experimental_datasets,
        mode="validation",
    )


if __name__ == "__main__":
    config = OmegaConf.load("conf/analysis.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
