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
    dataset_names = [ds_name for ds_name in configs[config.analysis.dir]['dataset']['name'].split('_')]
    clusters, silhouette_avg = hierarchical_clustering_algorithm(dataset_names, distance='correlation',
                                                                 method='ward', metric=None,
                                                                 truncate_mode='lastp', p=12,
                                                                 criterion='maxclust', criterion_value=4, verbose=False,
                                                                 )
    print('Silhouette average:', round(silhouette_avg, 4), "(quality of clustering)\n")

    group_by = 'four'
    ref_dict = load_reference(group_by=group_by)
    grouped_clusters = neuron_distribution(clusters, ref_dict, stat='percent', group_by=group_by, plot_type='both')

if __name__ == "__main__":
    config = OmegaConf.load("conf/analysis.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    analysis(config)
