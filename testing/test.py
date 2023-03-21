import hydra
from omegaconf import OmegaConf, DictConfig
import logging
import os

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="testing")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(os.getcwd())


if __name__ == "__main__":
    main()
