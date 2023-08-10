from data._utils import *

# Init logger
logger = logging.getLogger(__name__)

def get_datasets(dataset_config: DictConfig, name='training'):

    train_dataset, val_dataset = None, None

    

    experimental_datasets = dataset_config.experimental_datasets
    num_named_neurons = dataset_config.num_named_neurons
    num_worms = dataset_config.num_worms
    k_splits = dataset_config.k_splits
    num_train_samples = dataset_config.num_train_samples
    num_val_samples = dataset_config.num_val_samples
    seq_len = dataset_config.seq_len
    tau = dataset_config.tau
    reverse = dataset_config.reverse
    use_residual = dataset_config.use_residual
    smooth_data = dataset_config.smooth_data

    # Verifications
    assert isinstance(k_splits, int) and k_splits > 1, "k_splits must be an integer > 1"

    assert isinstance(num_named_neurons, int) or num_named_neurons == "all", (
        "num_named_neurons must be a positive integer or 'all'."
    )

    assert isinstance(num_worms, int) or num_worms == "all", (
        "num_worms must be a positive integer or 'all'."
    )

    combined_dataset, dataset_info, neuron_counts = create_combined_dataset(experimental_datasets,
                                                                            num_named_neurons, num_worms,
                                                                            name=name)
    train_dataset, val_dataset = split_combined_dataset(combined_dataset, k_splits, num_train_samples,
                                                         num_val_samples, seq_len, tau, reverse,
                                                         use_residual, smooth_data)
    
    # Save the datasets and information about them
    log_dir = os.getcwd()  # logs/hydra/${now:%Y_%m_%d_%H_%M_%S}
    os.makedirs(os.path.join(log_dir, 'dataset'), exist_ok=True)
    dataset_info.to_csv(os.path.join(log_dir, 'dataset', f"dataset_info_for_{name}.csv"), index=True, header=True)
    neuron_counts.to_csv(os.path.join(log_dir, 'dataset', f"neuron_info_for_{name}.csv"), index=True, header=True)
    torch.save(train_dataset, os.path.join(log_dir, 'dataset', f"train_dataset_for_{name}.pt"))
    torch.save(val_dataset, os.path.join(log_dir, 'dataset', f"val_dataset_for_{name}.pt"))

    if dataset_config.use_this_train_dataset is not None:
        logger.info(f'Overwiting train dataset with {dataset_config.use_this_train_dataset}')
        train_dataset = torch.load(dataset_config.use_this_train_dataset)
    if dataset_config.use_this_val_dataset is not None:
        logger.info(f'Overwiting val. dataset with {dataset_config.use_this_val_dataset}')
        val_dataset = torch.load(dataset_config.use_this_val_dataset)

    return train_dataset, val_dataset

if __name__ == "__main__":
    config = OmegaConf.load("configs/submodule/dataset.yaml")
    print(OmegaConf.to_yaml(config), end="\n\n")
    dataset_train = get_datasets(config.dataset.train)
    dataset_predict = get_datasets(config.dataset.predict)
