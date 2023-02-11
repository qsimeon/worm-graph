from data._utils import *


def get_dataset(config: DictConfig):
    """
    Returns a generator object that yields single worm data objects (dict)
    from the multi-worm dataset specified by the `name` param in 'dataset.yaml'.
    """
    # load the dataset
    dataset_name = config.dataset.name
    dataset = load_dataset(dataset_name)
    print(
        "Chosen dataset: {}\nNum. worms: {}\Generator: {}".format(
            dataset["dataset_name"],
            dataset["num_worms"],
            dataset["dataset_generator"],
        ),
        end="\n\n",
    )
    return dataset


if __name__ == "__main__":
    get_dataset(OmegaConf.load("conf/dataset.yaml"))
