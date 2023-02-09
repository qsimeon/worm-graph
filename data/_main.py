from data._utils import *


def get_dataset(config: DictConfig):
    """
    Returns a generator object that yields single worm data objects (dict)
    from the multi-worm dataset specified by the `name` param in 'dataset.yaml'.
    """
    # load the dataset
    dataset_name = config.name
    all_worms_dataset = load_dataset(dataset_name)
    print(
        "Chosen dataset: {}\nWorms: {}".format(dataset_name, list(all_worms_dataset)),
        end="\n\n",
    )
    single_worm_gen = iter(all_worms_dataset.items())
    # return generator for single worm datasets
    yield from single_worm_gen


if __name__ == "__main__":
    get_dataset(OmegaConf.load("conf/dataset.yaml"))
