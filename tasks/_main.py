from _utils import *


def create_task(config: DictConfig) -> None:
    print("Task parameter:", config.task.param, end="\n\n")
    return None


if __name__ == "__main__":
    config = OmegaConf.load("conf/task.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    create_task(config)
