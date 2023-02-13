from data._utils import *


def get_dataset(config: DictConfig):
    """
    Returns a generator object that yields single worm data objects (dict)
    from the multi-worm dataset specified by the `name` param in 'dataset.yaml'.
    """
    # load the dataset
    dataset_name = config.dataset.name
    multi_worms_dataset = load_dataset(dataset_name)
    print(
        "Chosen dataset: {}\nNum. worms: {}\nWorm names: {}".format(
            dataset_name,
            len(multi_worms_dataset),
            list(multi_worms_dataset.keys()),
        ),
        end="\n\n",
    )
    return multi_worms_dataset


if __name__ == "__main__":
    config = OmegaConf.load("conf/dataset.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    dataset = get_dataset(config)
