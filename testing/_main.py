from testing._utils import *


def test_config(config: DictConfig) -> None:
    # current working directory
    print("current working dir:\n\t", os.getcwd(), end="\n\n")
    # load the config
    og_cfg = OmegaConf.to_yaml(config)
    print("original config yaml:\n", og_cfg, end="\n\n")
    # modify the config
    new_cfg = OmegaConf.structured(OmegaConf.to_yaml(config))
    new_cfg.database = {"hostname": "database01", "port": 3306}
    new_cfg.train = None
    new_cfg.model = None
    # display the modified config
    print("modified config yaml:\n", OmegaConf.to_yaml(new_cfg), end="\n\n")
    


if __name__ == "__main__":
    config = OmegaConf.load("conf/test_config.yaml")
    print("\nconfig:\n\t", OmegaConf.to_yaml(config), end="\n\n")
    test_config(config)
