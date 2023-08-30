from analysis._utils import *

# Init logger
logger = logging.getLogger(__name__)

def analyse_run(analysis_config: DictConfig):

    log_dir = analysis_config.analyse_this_log_dir

    assert log_dir is not None, "log_dir is None. Please specify a log directory to analyse."

    # Analyse loss spread across datasets
    validation_loss_per_dataset(log_dir)

    #TODO Hierarchical Clustering analysis

if __name__ == "__main__":
    config = OmegaConf.load("conf/analysis.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
