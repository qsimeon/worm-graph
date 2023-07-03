#!/usr/bin/env python
# encoding: utf-8

from analysis._utils import *
from analysis._pkg import *


def analysis(
    config: DictConfig,
):
    configs = {}
    for file_path in find_config_files(config.analysis.dir):
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
            # Do something with the data
            configs[os.path.dirname(file_path)] = OmegaConf.create(data)

    # === Hierarchical clustering ===
    dataset_names = ['Uzel2022', 'Skora2018']#[ds_name for ds_name in configs[config.analysis.dir]['dataset']['name'].split('_')]
    (all_worm_clusters, ref_dict,
    count_inside_clusters_array,
    silhouettes) = hc_analyse_dataset(dataset_names, apply_suggestion=True, hip='hip1', group_by='four', method='ward',
                       metric=None)

if __name__ == "__main__":
    config = OmegaConf.load("conf/analysis.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    analysis(config)
