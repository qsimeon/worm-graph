from _utils import *


@hydra.main(version_base=None, config_path="configs", config_name="data")
def get_dataset(config):
    # load the dataset
    dataset_name = config.dataset.name
    all_worms_dataset = load_dataset(dataset_name)
    print("Chosen dataset:", dataset_name, end="\n\n")
    # pick one worm at random
    worm = np.random.choice(list(all_worms_dataset.keys()))
    single_worm_dataset = pick_worm(all_worms_dataset, worm)
    print("Picked:", worm, end="\n\n")
    return single_worm_dataset
