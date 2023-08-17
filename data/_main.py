from data._utils import *

# Init logger
logger = logging.getLogger(__name__)

def get_datasets(dataset_config: DictConfig):

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

    log_dir = os.getcwd()  # logs/hydra/${now:%Y_%m_%d_%H_%M_%S}
    os.makedirs(os.path.join(log_dir, 'dataset'), exist_ok=True)

    if dataset_config.use_these_datasets is not None:
        # Assert that the directory exists
        assert os.path.isdir(dataset_config.use_these_datasets), (
            f"Directory {dataset_config.use_these_datasets} does not exist."
        )
        
        # Flags to know if the datasets already exist
        train_dataset_exists = False
        val_dataset_exists = False

        # Check what files are in the directory
        ds_files = os.listdir(dataset_config.use_these_datasets)

        if 'train_dataset.pt' in ds_files:
            assert 'train_dataset_info.csv' in ds_files, (
                "train_dataset.pt exists but train_dataset_info.csv does not."
            )
            logger.info("Loading provided train dataset from %s" % dataset_config.use_these_datasets)
            train_dataset_exists = True
            train_dataset = torch.load(os.path.join(dataset_config.use_these_datasets, 'train_dataset.pt'))
            dataset_info_train = pd.read_csv(os.path.join(dataset_config.use_these_datasets, 'train_dataset_info.csv'), index_col=0)

        if 'val_dataset.pt' in ds_files:
            assert 'val_dataset_info.csv' in ds_files, (
                "val_dataset.pt exists but val_dataset_info.csv does not."
            )
            logger.info("Loading provided validation dataset from %s" % dataset_config.use_these_datasets)
            val_dataset_exists = True
            val_dataset = torch.load(os.path.join(dataset_config.use_these_datasets, 'val_dataset.pt'))
            dataset_info_val = pd.read_csv(os.path.join(dataset_config.use_these_datasets, 'val_dataset_info.csv'), index_col=0)

        # If both datasets exist, save and return them. Else, create them using the experimental datasets
        if train_dataset_exists and val_dataset_exists:
            # There's no need to save the datasets again, just return them
            return train_dataset, val_dataset
        
        else:
            # Logging info
            if not train_dataset_exists:
                logger.info("No train dataset found in %s, creating a new one" % dataset_config.use_these_datasets)
            if not val_dataset_exists:
                logger.info("No validation dataset found in %s, creating a new one" % dataset_config.use_these_datasets)

            # Check if combined dataset was provided. If so, load it, otherwise create it from the experimental datasets
            if 'combined_dataset.pickle' in ds_files:
                logger.info("Using provided combined dataset from %s to create the missing datasets" % dataset_config.use_these_datasets)
                with open(os.path.join(dataset_config.use_these_datasets, 'combined_dataset.pickle'), 'rb') as f:
                    combined_dataset = pickle.load(f)
                combined_dataset, dataset_info = filter_loaded_combined_dataset(combined_dataset, num_worms, num_named_neurons)
                
            else:
                logger.info("Creating combined dataset from experimental datasets")
                combined_dataset, dataset_info = create_combined_dataset(experimental_datasets,
                                                                         num_named_neurons, num_worms
                                                                         )
                
            created_train_dataset, created_val_dataset, time_step_info = split_combined_dataset(
                    combined_dataset, k_splits, num_train_samples,
                    num_val_samples, seq_len, tau, reverse,
                    use_residual, smooth_data
                )
            
            # Merge dataset_info and time_step_info
            created_dataset_info_train = dataset_info.merge(time_step_info[['combined_dataset_index', 'train_time_steps']], on='combined_dataset_index', how='outer')
            created_dataset_info_val = dataset_info.merge(time_step_info[['combined_dataset_index', 'val_time_steps']], on='combined_dataset_index', how='outer')

            # Replace missing dataset
            if not train_dataset_exists:
                train_dataset = created_train_dataset
                dataset_info_train = created_dataset_info_train
            if not val_dataset_exists:
                val_dataset = created_val_dataset
                dataset_info_val = created_dataset_info_val

            # Save the datasets and their information (just train and validation, no need to save combined)
            dataset_info_train.to_csv(os.path.join(log_dir, 'dataset', f"train_dataset_info.csv"), index=True, header=True)
            dataset_info_val.to_csv(os.path.join(log_dir, 'dataset', f"val_dataset_info.csv"), index=True, header=True)

            torch.save(train_dataset, os.path.join(log_dir, 'dataset', f"train_dataset.pt"))
            torch.save(val_dataset, os.path.join(log_dir, 'dataset', f"val_dataset.pt"))

            return train_dataset, val_dataset
            
    else:
        # Create the datasets using the experimental datasets
        logger.info("Creating validation and train datasets from experimental datasets")
        combined_dataset, dataset_info = create_combined_dataset(experimental_datasets,
                                                                 num_named_neurons, num_worms
                                                                 )
        train_dataset, val_dataset, time_step_info = split_combined_dataset(combined_dataset, k_splits, num_train_samples,
                                                                            num_val_samples, seq_len, tau, reverse,
                                                                            use_residual, smooth_data
                                                                            )
        
        # Merge dataset_info and time_step_info
        dataset_info_train = dataset_info.merge(time_step_info[['combined_dataset_index', 'train_time_steps']], on='combined_dataset_index', how='outer')
        dataset_info_val = dataset_info.merge(time_step_info[['combined_dataset_index', 'val_time_steps']], on='combined_dataset_index', how='outer')

        # Save the datasets and information about them
        # => They contain the same neurons, but with different time steps
        dataset_info.to_csv(os.path.join(log_dir, 'dataset', f"combined_dataset_info.csv"), index=True, header=True)
        dataset_info_train.to_csv(os.path.join(log_dir, 'dataset', f"train_dataset_info.csv"), index=True, header=True)
        dataset_info_val.to_csv(os.path.join(log_dir, 'dataset', f"val_dataset_info.csv"), index=True, header=True)

        torch.save(train_dataset, os.path.join(log_dir, 'dataset', f"train_dataset.pt"))
        torch.save(val_dataset, os.path.join(log_dir, 'dataset', f"val_dataset.pt"))
        with open(os.path.join(log_dir, 'dataset', f"combined_dataset.pickle"), 'wb') as f:
            pickle.dump(combined_dataset, f)

        return train_dataset, val_dataset

if __name__ == "__main__":
    config = OmegaConf.load("configs/submodule/dataset.yaml")
    print(OmegaConf.to_yaml(config), end="\n\n")
    dataset_train = get_datasets(config.dataset.train)
    dataset_predict = get_datasets(config.dataset.predict)
