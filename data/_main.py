from data._utils import *


def get_dataset(config: DictConfig):
    """
    Returns a generator object that yields single worm data objects (dict)
    from the multi-worm dataset specified by the `name` param in 'dataset.yaml'.
    """
    # modified to combine datasets when given list
    if isinstance(config.dataset.name, str):
        dataset_names = [config.dataset.name]
    else:
        dataset_names = sorted(list(config.dataset.name))
    if isinstance(config.smooth.name, str):
        smooth_method = [config.smooth.name]
    else:
        smooth_method = sorted(list(config.smooth.name))
    # load the dataset(s)
    combined_dataset = dict()
    for dataset_name in dataset_names:
        multi_worms_dataset = load_dataset(dataset_name)
        for worm in multi_worms_dataset:
            if worm in combined_dataset:
                worm_ = "worm%s" % len(combined_dataset)
                combined_dataset[worm_] = multi_worms_dataset[worm]
                combined_dataset[worm_]["worm"] = worm_
                combined_dataset[worm_]["dataset"] = "_".join(dataset_names)
            else:
                combined_dataset[worm] = multi_worms_dataset[worm]
                combined_dataset[worm]["dataset"] = "_".join(dataset_names)
    # display the dataset
    print(
        "Chosen dataset(s): {}\nNum. worms: {}\nWorm names: {}\nSmoothing Method:{}".format(
            dataset_names,
            len(combined_dataset),
            list(combined_dataset.keys()),
            smooth_method,
        ),
        end="\n\n",
    )

    return combined_dataset


if __name__ == "__main__":
    root = os.path.abspath(os.path.dirname(os.getcwd()))
    root += "/"
    config = OmegaConf.load(root + "conf/dataset.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    dataset = get_dataset(config)
