from data._utils import *


def get_dataset(config: DictConfig):
    """Returns a dict with the worm data of all requested datasets.

    Returns a generator object that yields single worm data objects (dict)
    from the multi-worm dataset specified by the `name` param in 'dataset.yaml'.

    Parameters
    ----------
    config: DictConfig
        Hydra configuration object.

    Calls
    -----
    load_dataset : function in data/_utils.py
        Load a specified dataset by name.

    Returns
    -------
    dataset: dict
        Dictionary of single worm data objects.

    Notes
    -----
    * The keys of the dictionary are the worm IDs ('worm0', 'worm1', etc.).
    * The features of each worm are (22 in total):
        'calcium_data', 'dataset', 'dt', 'max_timesteps', 'named_neuron_to_slot',
        'named_neurons_mask', 'neuron_to_slot', 'neurons_mask',
        'num_named_neurons', 'num_neurons', 'num_unknown_neurons',
        'residual_calcium', 'slot_to_named_neuron', 'slot_to_neuron',
        'slot_to_unknown_neuron', 'smooth_calcium_data', 'smooth_method',
        'smooth_residual_calcium', 'time_in_seconds', 'unknown_neuron_to_slot',
        'unknown_neurons_mask', 'worm'.
    """

    # TODO: Add option to the `conf/data.yaml` config to leave out certain worms from dataset
    # Modified to combine datasets when given list
    if isinstance(config.dataset.name, str):
        dataset_names = [config.dataset.name]
    else:
        dataset_names = sorted(list(config.dataset.name))

    # Load the dataset(s)
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

    # Display the dataset
    print(
        "Chosen dataset(s): {}\nNum. worms: {}".format(
            dataset_names,
            len(combined_dataset),
        ),
        end="\n\n",
    )

    return combined_dataset


if __name__ == "__main__":
    config = OmegaConf.load("conf/dataset.yaml")
    print("\nconfig:\n\t", OmegaConf.to_yaml(config), end="\n\n")
    dataset = get_dataset(config)
