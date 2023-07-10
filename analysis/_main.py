from analysis._utils import *
from analysis._pkg import *


def analysis(
    config: DictConfig,
):
    # Combine datasets when given a list of dataset names
    if isinstance(config.analysis.dataset_name, str):
        dataset_names = [config.analysis.dataset_name]
    else:
        dataset_names = sorted(list(config.analysis.dataset_name))

    # Perform hierarchical clustering
    (all_worm_clusters, ref_dict, silhouettes) = hc_analyse_dataset(
        dataset_names,
        apply_suggestion=True,
        hip="hip1",
        group_by=config.analysis.hierarchical_clustering.group_by,
        method=config.analysis.hierarchical_clustering.method,
        metric=config.analysis.hierarchical_clustering.metric,
    )


if __name__ == "__main__":
    config = OmegaConf.load("conf/analysis.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    analysis(config)
