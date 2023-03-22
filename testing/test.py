import os
import hydra
from omegaconf import OmegaConf, DictConfig

@hydra.main(version_base=None, config_path="../conf", config_name="testing")
def main(cfg: DictConfig) -> None:
    print("Working directory:", os.getcwd(), end="\n\n")
    print("`config` YAML:\n", OmegaConf.to_yaml(cfg), end="\n\n")


if __name__ == "__main__":
    main()
