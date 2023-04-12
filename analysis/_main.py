#!/usr/bin/env python
# encoding: utf-8

from analysis._utils import *
from analysis._pkg import *

def analysis(config: DictConfig,):
    configs = {}
    for file_path in find_config_files(config.analysis.dir):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            # Do something with the data
            configs[os.path.dirname(file_path)] = OmegaConf.create(data)
    

    # import pdb; pdb.set_trace()
    # print(get_config_value(configs['../logs/hydra/2023_03_30_16_59_28'], 'preprocess.smooth'))
    # print(get_config_value(configs['../logs/hydra/2023_03_30_16_59_28'], 'model.type'))

    # print(os.getcwd())
    # print(type(configs), len(configs))

    # d = configs['logs/hydra/2023_04_03_20_53_10/dataset.name=Uzel2022,globals.shuffle=True,train.epochs=50']
    # print(type(d))
    # print(OmegaConf.to_yaml(d))

    plot_loss_vs_parameter_sorted(configs, param_names=config.analysis.param_names)


    # plotting predictions
    for log_dir in configs:
        with open(os.path.join(log_dir + "/config.yaml"), 'r') as f:
            yaml_file = yaml.safe_load(f)
            plot_targets_predictions(
                log_dir,
                "worm0",
                "AVAL",
                get_config_value(yaml_file, 'global.use_residuals'),
            )



if __name__ == "__main__":
    config = OmegaConf.load("conf/analysis.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    analysis(config)
