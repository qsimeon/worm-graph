import os
import hydra
from omegaconf import OmegaConf, DictConfig, open_dict


@hydra.main(config_path="../conf", config_name="test_config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("orig config yaml:\n", OmegaConf.to_yaml(cfg), end="\n\n")
    new_cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    new_cfg.database = {"hostname": "database01", "port": 3306}
    new_cfg.train = None
    new_cfg.model = None
    print("new config yaml:\n", OmegaConf.to_yaml(new_cfg), end="\n\n")
    OmegaConf.save(new_cfg, os.path.join(os.getcwd(), "cfg.yaml"))
    print("working dir:\n\t", os.getcwd(), end="\n\n")


if __name__ == "__main__":
    main()
