from data._utils import *

# Init logger
logger = logging.getLogger(__name__)

def get_dataset(dataset_config: DictConfig):
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

    # TODO: Add option to the `conf/dataset.yaml` config to leave out certain worms from dataset
    # Combine datasets when given a list of dataset names
    if isinstance(dataset_config.name, str):
        dataset_names = [dataset_config.name]
    else:
        dataset_names = sorted(list(dataset_config.name))

    num_named_neurons = dataset_config.num_named_neurons
    num_worms = dataset_config.num_worms

    # Assert num_named_neurons is a positive integer or 'all'
    assert isinstance(num_named_neurons, int) or num_named_neurons == "all", (
        "num_named_neurons must be a positive integer or 'all'."
    )

    # Assert num_worms is a positive integer or 'all'
    assert isinstance(num_worms, int) or num_worms == "all", (
        "num_worms must be a positive integer or 'all'."
    )

    # Load the dataset(s)
    combined_dataset = dict()
    for dataset_name in dataset_names:
        logger.info('Loading single dataset: {}'.format(dataset_name))
        multi_worms_dataset = load_dataset(dataset_name)

        # Select the `num_named_neurons` neurons (overwrite the masks)
        if num_named_neurons != "all":
            multi_worms_dataset = select_named_neurons(multi_worms_dataset, num_named_neurons)

        for worm in multi_worms_dataset:
            if worm in combined_dataset:
                worm_ = "worm%s" % len(combined_dataset)
                combined_dataset[worm_] = multi_worms_dataset[worm]
                combined_dataset[worm_]["worm"] = worm_
                combined_dataset[worm_]["dataset"] = "_".join(dataset_names)
            else:
                combined_dataset[worm] = multi_worms_dataset[worm]
                combined_dataset[worm]["dataset"] = "_".join(dataset_names)

    # Verify if len(combined_dataset) is >= num_worms
    if num_worms != "all":
        assert len(combined_dataset) >= num_worms, (
            "num_worms must be less than or equal to the number of worms in the dataset(s). "
        )

        # Select `num_worms` worms
        wormIDs = [wormID for wormID in combined_dataset.keys()]
        wormIDs_to_keep = np.random.choice(wormIDs, size=num_worms, replace=False)
        logger.info('Selecting {} worms from {}'.format(len(wormIDs_to_keep), len(combined_dataset)))

        # Remove the worms that are not in `wormIDs_to_keep`
        for wormID in wormIDs:
            if wormID not in wormIDs_to_keep:
                combined_dataset.pop(wormID)

    combined_dataset = {f"worm{i}": combined_dataset[key] for i, key in enumerate(combined_dataset.keys())}

    available_neurons = []
    for worm, data in combined_dataset.items():
        available_neurons.append([neuron for slot, neuron in data['slot_to_named_neuron'].items()])

    logger.info('Available neurons: {}'.format(available_neurons))

    logger.debug('Combined dataset loaded: {}'.format(dataset_names))

    # Save the dataset
    if dataset_config.save:
        file = os.path.join(ROOT_DIR, "data/processed/neural", "custom.pickle")
        with open(file, "wb") as f:
            pickle.dump(combined_dataset, f)

    return combined_dataset


if __name__ == "__main__":
    config = OmegaConf.load("configs/submodule/dataset.yaml")
    print(OmegaConf.to_yaml(config), end="\n\n")
    dataset_train = get_dataset(config.dataset.train)
    dataset_predict = get_dataset(config.dataset.predict)
