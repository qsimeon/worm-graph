from _utils import *


def create_task(config: DictConfig) -> None:
    print("Task parameter:", config.task.param, end="\n\n")
    return None


if __name__ == "__main__":
    create_task(OmegaConf.load("conf/task.yaml"))
