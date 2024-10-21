from data._utils import *

# Init logger
logger = logging.getLogger(__name__)


def get_datasets(data_config: DictConfig, save=False):
    """Retrieve or generate training and validation sets based on the provided configuration.

    The function first checks if datasets are provided in the specified directory. If they are,
    it will load them. If not, it will generate the datasets from the requested source datasets.

    Params
    ------
    data_config: (DictConfig)
        A configuration dictionary containing the following keys:

        - use_these_datasets (str or None): Directory containing train and validation datasets.
            If provided, datasets will be loaded from this directory if they exist.
            If not provided or datasets don't exist in the directory,
            they will be generated using the parameters below.
        - source_datasets (str): Path to the source datasets.
        - num_labeled_neurons (int or None): Number of labeled neurons to include or None to include all.
        - num_worms (int or None): Number of worms to include or None to include all.
        - num_train_samples (int): Number of training samples per worm.
        - num_val_samples (int): Number of validation samples per worm.
        - seq_len (int): Sequence length for time series data.
        - reverse (bool): Whether to reverse the time series data.
        - use_residual (bool): Whether to use residuals in the data.
        - use_smooth (bool): Whether to smooth the data.

    save: (bool)
        Whether to save the datasets and their information to the log directory.
        Setting not available to modify in the config file (just for devs).

    Returns
    -------
    tuple: A tuple containing two elements:
        - train_dataset (torch.Tensor): The training dataset.
        - val_dataset (torch.Tensor): The validation dataset.
    """
    # Parse out parameters from the config
    source_datasets = data_config.source_datasets
    num_labeled_neurons = data_config.num_labeled_neurons
    num_train_samples = data_config.num_train_samples
    num_val_samples = data_config.num_val_samples
    seq_len = data_config.seq_len
    reverse = data_config.reverse
    use_residual = data_config.use_residual
    use_smooth = data_config.use_smooth
    train_split_first = data_config.train_split_first
    train_split_ratio = data_config.train_split_ratio
    save = data_config.save_datasets or save
    # The dataset containing all the available source datasets was presaved
    presave_path = None
    all_experiment = all(
        [
            (dataset in source_datasets and source_datasets[dataset] == "all")
            for dataset in EXPERIMENT_DATASETS
        ]
    )
    if all_experiment:
        logger.info(
            f"Requested dataset pattern matched the presaved dataset.\n"
            f"Setting `config.use_these_datasets.path` to the presaved directory.\n\n"
        )
        presave_path = os.path.join(ROOT_DIR, "data", "datasets")
        os.makedirs(presave_path, exist_ok=True)
        # Check if the directory is not empty
        if os.path.isdir(presave_path) and os.listdir(presave_path):
            data_config.use_these_datasets.path = presave_path
        else:  # Go to (*)
            logger.info(
                f"Directory {presave_path} is empty.\n"
                f"Creating the datasets from source files.\n\n"
            )
    # Initialize datasets
    full_dataset, train_dataset, val_dataset = None, None, None
    # Verifications
    assert isinstance(num_labeled_neurons, int) or (
        num_labeled_neurons is None
    ), "`num_labeled_neurons` must be a positive integer or None."
    # Control flow for loading or creating datasets
    if data_config.use_these_datasets.path is not None:
        # Assert that the directory exists
        assert os.path.isdir(
            data_config.use_these_datasets.path
        ), f"Directory {data_config.use_these_datasets.path} does not exist."
        # Flags to know if the datasets already exist
        full_dataset_exists = False
        train_dataset_exists = False
        val_dataset_exists = False
        # Check what files are in the directory
        ds_files = os.listdir(data_config.use_these_datasets.path)
        if "full_dataset.pt" in ds_files:
            assert (
                "full_dataset_info.csv" in ds_files
            ), "full_dataset.pt exists but full_dataset_info.csv does not."
            logger.info(
                "Loading provided full dataset from %s." % data_config.use_these_datasets.path
            )
            full_dataset_exists = True
            full_dataset = torch.load(
                os.path.join(data_config.use_these_datasets.path, "full_dataset.pt")
            )
            dataset_info_full = pd.read_csv(
                os.path.join(data_config.use_these_datasets.path, "full_dataset_info.csv"),
                index_col=0,
                converters={"neurons": ast.literal_eval},
            )
        if "train_dataset.pt" in ds_files:
            assert (
                "train_dataset_info.csv" in ds_files
            ), "train_dataset.pt exists but train_dataset_info.csv does not."
            logger.info(
                "Loading provided train dataset from %s." % data_config.use_these_datasets.path
            )
            train_dataset_exists = True
            train_dataset = torch.load(
                os.path.join(data_config.use_these_datasets.path, "train_dataset.pt")
            )
            dataset_info_train = pd.read_csv(
                os.path.join(data_config.use_these_datasets.path, "train_dataset_info.csv"),
                index_col=0,
                converters={"neurons": ast.literal_eval},
            )
        if "val_dataset.pt" in ds_files:
            assert (
                "val_dataset_info.csv" in ds_files
            ), "val_dataset.pt exists but val_dataset_info.csv does not."
            logger.info(
                "Loading provided validation dataset from %s." % data_config.use_these_datasets.path
            )
            val_dataset_exists = True
            val_dataset = torch.load(
                os.path.join(data_config.use_these_datasets.path, "val_dataset.pt")
            )
            dataset_info_val = pd.read_csv(
                os.path.join(data_config.use_these_datasets.path, "val_dataset_info.csv"),
                index_col=0,
                converters={"neurons": ast.literal_eval},
            )
        # If full, train and val data present, load and return them
        if full_dataset_exists and train_dataset_exists and val_dataset_exists and save:
            # Just save the dataset information (for use in the visualization submodule)
            dataset_info_full.to_csv(
                os.path.join(presave_path, f"full_dataset_info.csv"),
                index=True,
                header=True,
            )
            dataset_info_train.to_csv(
                os.path.join(presave_path, f"train_dataset_info.csv"),
                index=True,
                header=True,
            )
            dataset_info_val.to_csv(
                os.path.join(presave_path, f"val_dataset_info.csv"),
                index=True,
                header=True,
            )
            return full_dataset, train_dataset, val_dataset
        # Otherwise create them using the source datasets
        else:
            # Logging info
            if not full_dataset_exists:
                logger.info(
                    "No full dataset found in %s, creating a new one."
                    % data_config.use_these_datasets.path
                )
            if not train_dataset_exists:
                logger.info(
                    "No train dataset found in %s, creating a new one."
                    % data_config.use_these_datasets.path
                )
            if not val_dataset_exists:
                logger.info(
                    "No validation dataset found in %s, creating a new one."
                    % data_config.use_these_datasets.path
                )
            # Check if combined_dataset.pickle file present
            if "combined_dataset.pickle" in ds_files:
                logger.info(
                    "Creating datasets from combined_dataset.pick found in %s."
                    % data_config.use_these_datasets.path
                )
                with open(
                    os.path.join(
                        data_config.use_these_datasets.path,
                        "combined_dataset.pickle",
                    ),
                    "rb",
                ) as f:
                    combined_dataset_dict = pickle.load(f)
                combined_dataset_dict, dataset_info_full = filter_loaded_combined_dataset(
                    combined_dataset_dict,
                    data_config.use_these_datasets.num_worms,
                    num_labeled_neurons,
                )
            else:
                logger.info("Creating combined dataset from individual source datasets.")
                combined_dataset_dict, dataset_info_full = create_combined_dataset(
                    source_datasets, num_labeled_neurons
                )
            # Use largest `seq_len` that produces required unique samples from shortest dataset
            if seq_len is None:
                max_num_samples = max(num_train_samples, num_val_samples)
                min_timesteps = min(
                    dataset["max_timesteps"] for _, dataset in combined_dataset_dict.items()
                )
                seq_len = (min_timesteps // 2) - max_num_samples - 1
            logger.info(f"Chosen sequence length: {seq_len}\n")
            # Get full, train and validation sets from the combined dataset
            full_dataset, train_dataset, val_dataset, dataset_info_split = split_combined_dataset(
                combined_dataset_dict,
                num_train_samples,
                num_val_samples,
                seq_len,
                reverse,
                use_residual,
                use_smooth,
                train_split_first,
                train_split_ratio,
            )
            if any(ds is None for ds in [full_dataset, train_dataset, val_dataset]):
                raise ValueError(
                    f"Error creating dataset. No sequences of length {seq_len} could be sampled."
                )
            # Merge dataset_info_full and dataset_info_full
            dataset_info_train = dataset_info_full.merge(
                dataset_info_split[
                    [
                        "full_dataset_index",
                        "train_time_steps",
                        "num_train_samples",
                        "train_seq_len",
                        "train_split_idx",
                        "use_smooth",
                        "use_residual",
                        "train_split_first",
                        "train_split_ratio",
                    ]
                ],
                on="full_dataset_index",
                how="outer",
            )
            dataset_info_val = dataset_info_full.merge(
                dataset_info_split[
                    [
                        "full_dataset_index",
                        "val_time_steps",
                        "num_val_samples",
                        "val_seq_len",
                        "val_split_idx",
                        "use_smooth",
                        "use_residual",
                        "train_split_first",
                        "train_split_ratio",
                    ]
                ],
                on="full_dataset_index",
                how="outer",
            )
            # Save the dataset .pt files
            if save:
                torch.save(full_dataset, os.path.join(presave_path, f"full_dataset.pt"))
                torch.save(train_dataset, os.path.join(presave_path, f"train_dataset.pt"))
                torch.save(val_dataset, os.path.join(presave_path, f"val_dataset.pt"))
            # (Re)save the dataset information regardless of `save` argument
            # => full, train and val. sets contain the same neurons, but with different time steps and other info
            dataset_info_full.to_csv(
                os.path.join(presave_path, f"full_dataset_info.csv"),
                index=True,
                header=True,
            )
            dataset_info_train.to_csv(
                os.path.join(presave_path, f"train_dataset_info.csv"),
                index=True,
                header=True,
            )
            dataset_info_val.to_csv(
                os.path.join(presave_path, f"val_dataset_info.csv"),
                index=True,
                header=True,
            )
            return full_dataset, train_dataset, val_dataset
    else:  # (*)
        # Create the datasets using the source datasets
        logger.info("Creating full, train and validation sets from source files.")
        combined_dataset_dict, dataset_info_full = create_combined_dataset(
            source_datasets, num_labeled_neurons
        )
        # Use largest seq_len that produce num. unique samples from shortest dataset
        if seq_len is None:
            max_num_samples = max(num_train_samples, num_val_samples)
            min_timesteps = min(
                (dataset["max_timesteps"] for _, dataset in combined_dataset_dict.items())
            )
            seq_len = (min_timesteps // 2) - max_num_samples - 1
        logger.info(f"Chosen sequence length: {seq_len}\n.")
        # Get full, train and validation sets from the combined dataset
        full_dataset, train_dataset, val_dataset, dataset_info_split = split_combined_dataset(
            combined_dataset_dict,
            num_train_samples,
            num_val_samples,
            seq_len,
            reverse,
            use_residual,
            use_smooth,
            train_split_first,
            train_split_ratio,
        )
        if any(ds is None for ds in [full_dataset, train_dataset, val_dataset]):
            raise ValueError(
                f"Error creating dataset. No sequences of length {seq_len} could be sampled."
            )
        # Merge dataset_info_full and dataset_info_split
        dataset_info_train = dataset_info_full.merge(
            dataset_info_split[
                [
                    "full_dataset_index",
                    "train_time_steps",
                    "num_train_samples",
                    "train_seq_len",
                    "train_split_idx",
                    "use_smooth",
                    "use_residual",
                    "train_split_first",
                    "train_split_ratio",
                ]
            ],
            on="full_dataset_index",
            how="outer",
        )
        dataset_info_val = dataset_info_full.merge(
            dataset_info_split[
                [
                    "full_dataset_index",
                    "val_time_steps",
                    "num_val_samples",
                    "val_seq_len",
                    "val_split_idx",
                    "use_smooth",
                    "use_residual",
                    "train_split_first",
                    "train_split_ratio",
                ]
            ],
            on="full_dataset_index",
            how="outer",
        )
        # Delete the combined dataset column after merging (since it is not necessary anymore)
        dataset_info_train.drop(columns=["full_dataset_index"], inplace=True)
        dataset_info_val.drop(columns=["full_dataset_index"], inplace=True)
        # Save the dataset .pt files
        if save:
            torch.save(full_dataset, os.path.join(presave_path, f"full_dataset.pt"))
            torch.save(train_dataset, os.path.join(presave_path, f"train_dataset.pt"))
            torch.save(val_dataset, os.path.join(presave_path, f"val_dataset.pt"))
            with open(os.path.join(presave_path, f"combined_dataset.pickle"), "wb") as f:
                pickle.dump(combined_dataset_dict, f)
        # Save the dataset split information regardless
        # => train and val. datasets contain the same neurons, but with different time steps and other information
        dataset_info_full.to_csv(
            os.path.join(presave_path, f"full_dataset_info.csv"),
            index=True,
            header=True,
        )
        dataset_info_train.to_csv(
            os.path.join(presave_path, f"train_dataset_info.csv"),
            index=True,
            header=True,
        )
        dataset_info_val.to_csv(
            os.path.join(presave_path, f"val_dataset_info.csv"),
            index=True,
            header=True,
        )

        # Return the train and validation datasets
        return full_dataset, train_dataset, val_dataset


if __name__ == "__main__":
    config = OmegaConf.load("configs/submodule/data.yaml")
    print(OmegaConf.to_yaml(config), end="\n\n")
    dataset = get_datasets(config.dataset, save=True)
    print(f"full dataset: {type(dataset[0])}\n")
    print(f"train dataset: {type(dataset[1])}\n")
    print(f"validation dataset: {type(dataset[2])}\n")
