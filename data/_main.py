from data._utils import *

# Init logger
logger = logging.getLogger(__name__)


def get_datasets(dataset_config: DictConfig, save=True):
    """
    Retrieve or generate training and validation datasets based on the provided configuration.

    The function first checks if datasets are provided in the specified directory. If they are,
    it will load them. If not, it will generate the datasets from the requested experimental datasets.

    Params
    ------
    dataset_config: (DictConfig)
        A configuration dictionary containing the following keys:

        - use_these_datasets (str or None): Directory containing train and validation datasets.
            If provided, datasets will be loaded from this directory if they exist.
            If not provided or datasets don't exist in the directory, they will be generated using the other parameters.

        - experimental_datasets: Path to the experimental datasets.
        - num_named_neurons (int or 'all'): Number of named neurons to include or 'all' to include all.
        - num_worms (int or 'all'): Number of worms to include or 'all' to include all.
        - num_train_samples (int): Number of training samples.
        - num_val_samples (int): Number of validation samples.
        - seq_len (int): Sequence length for time series data.
        - reverse (bool): Whether to reverse the time series data.
        - use_residual (bool): Whether to use residuals in the data.
        - smooth_data (bool): Whether to smooth the data.

    save: (bool)
        Whether to save the datasets and their information to the log directory.
        Setting not available to modify in the config file (just for devs).

    Returns
    -------
    tuple: A tuple containing two elements:
        - train_dataset (torch.Tensor): The training dataset.
        - val_dataset (torch.Tensor): The validation dataset.
    """

    train_dataset, val_dataset = None, None

    experimental_datasets = dataset_config.experimental_datasets
    num_named_neurons = dataset_config.num_named_neurons
    num_train_samples = dataset_config.num_train_samples
    num_val_samples = dataset_config.num_val_samples
    seq_len = dataset_config.seq_len
    reverse = dataset_config.reverse
    use_residual = dataset_config.use_residual
    smooth_data = dataset_config.smooth_data

    # Verifications
    assert (
        isinstance(num_named_neurons, int) or num_named_neurons == "all"
    ), "num_named_neurons must be a positive integer or 'all'."

    log_dir = os.getcwd()  # logs/hydra/${now:%Y_%m_%d_%H_%M_%S}
    os.makedirs(os.path.join(log_dir, "dataset"), exist_ok=True)

    if dataset_config.use_these_datasets.path is not None:
        # Assert that the directory exists
        assert os.path.isdir(
            dataset_config.use_these_datasets.path
        ), f"Directory {dataset_config.use_these_datasets.path} does not exist."

        # Flags to know if the datasets already exist
        train_dataset_exists = False
        val_dataset_exists = False

        # Check what files are in the directory
        ds_files = os.listdir(dataset_config.use_these_datasets.path)

        if "train_dataset.pt" in ds_files:
            assert (
                "train_dataset_info.csv" in ds_files
            ), "train_dataset.pt exists but train_dataset_info.csv does not."
            logger.info(
                "Loading provided train dataset from %s"
                % dataset_config.use_these_datasets.path
            )
            train_dataset_exists = True
            train_dataset = torch.load(
                os.path.join(dataset_config.use_these_datasets.path, "train_dataset.pt")
            )
            dataset_info_train = pd.read_csv(
                os.path.join(
                    dataset_config.use_these_datasets.path, "train_dataset_info.csv"
                ),
                index_col=0,
            )

        if "val_dataset.pt" in ds_files:
            assert (
                "val_dataset_info.csv" in ds_files
            ), "val_dataset.pt exists but val_dataset_info.csv does not."
            logger.info(
                "Loading provided validation dataset from %s"
                % dataset_config.use_these_datasets.path
            )
            val_dataset_exists = True
            val_dataset = torch.load(
                os.path.join(dataset_config.use_these_datasets.path, "val_dataset.pt")
            )
            dataset_info_val = pd.read_csv(
                os.path.join(
                    dataset_config.use_these_datasets.path, "val_dataset_info.csv"
                ),
                index_col=0,
            )

        # If both datasets exist, save and return them. Else, create them using the experimental datasets
        if train_dataset_exists and val_dataset_exists:
            # There's no need to save the datasets again, just their information (for visualization submodule use)
            dataset_info_train.to_csv(
                os.path.join(log_dir, "dataset", f"train_dataset_info.csv"),
                index=True,
                header=True,
            )
            dataset_info_val.to_csv(
                os.path.join(log_dir, "dataset", f"val_dataset_info.csv"),
                index=True,
                header=True,
            )
            return train_dataset, val_dataset

        else:
            # Logging info
            if not train_dataset_exists:
                logger.info(
                    "No train dataset found in %s, creating a new one"
                    % dataset_config.use_these_datasets.path
                )
            if not val_dataset_exists:
                logger.info(
                    "No validation dataset found in %s, creating a new one"
                    % dataset_config.use_these_datasets.path
                )

            # Check if combined dataset was provided. If so, load it, otherwise create it from the experimental datasets
            if "combined_dataset.pickle" in ds_files:
                logger.info(
                    "Using provided combined dataset from %s to create the missing datasets"
                    % dataset_config.use_these_datasets.path
                )
                with open(
                    os.path.join(
                        dataset_config.use_these_datasets.path,
                        "combined_dataset.pickle",
                    ),
                    "rb",
                ) as f:
                    combined_dataset = pickle.load(f)
                combined_dataset, dataset_info = filter_loaded_combined_dataset(
                    combined_dataset,
                    dataset_config.use_these_datasets.num_worms,
                    num_named_neurons,
                )

            else:
                logger.info("Creating combined dataset from experimental datasets")
                combined_dataset, dataset_info = create_combined_dataset(
                    experimental_datasets, num_named_neurons
                )

            (
                created_train_dataset,
                created_val_dataset,
                dataset_info_split,
            ) = split_combined_dataset(
                combined_dataset,
                num_train_samples,
                num_val_samples,
                seq_len,
                reverse,
                use_residual,
                smooth_data,
            )

            # Merge dataset_info and time_step_info
            created_dataset_info_train = dataset_info.merge(
                dataset_info_split[
                    [
                        "combined_dataset_index",
                        "train_time_steps",
                        "num_train_samples",
                        "train_seq_len",
                        "smooth_data",
                        "use_residual",
                    ]
                ],
                on="combined_dataset_index",
                how="outer",
            )
            created_dataset_info_val = dataset_info.merge(
                dataset_info_split[
                    [
                        "combined_dataset_index",
                        "val_time_steps",
                        "num_val_samples",
                        "val_seq_len",
                        "smooth_data",
                        "use_residual",
                    ]
                ],
                on="combined_dataset_index",
                how="outer",
            )

            # Replace missing dataset
            if not train_dataset_exists:
                train_dataset = created_train_dataset
                dataset_info_train = created_dataset_info_train
            if not val_dataset_exists:
                val_dataset = created_val_dataset
                dataset_info_val = created_dataset_info_val

            # Save the datasets and their information (just train and validation, no need to save combined)
            dataset_info_train.to_csv(
                os.path.join(log_dir, "dataset", f"train_dataset_info.csv"),
                index=True,
                header=True,
            )
            dataset_info_val.to_csv(
                os.path.join(log_dir, "dataset", f"val_dataset_info.csv"),
                index=True,
                header=True,
            )

            if save:
                torch.save(
                    train_dataset, os.path.join(log_dir, "dataset", f"train_dataset.pt")
                )
                torch.save(
                    val_dataset, os.path.join(log_dir, "dataset", f"val_dataset.pt")
                )

            return train_dataset, val_dataset

    else:
        # Create the datasets using the experimental datasets
        logger.info("Creating validation and train datasets from experimental datasets")
        combined_dataset, dataset_info = create_combined_dataset(
            experimental_datasets, num_named_neurons
        )
        train_dataset, val_dataset, dataset_info_split = split_combined_dataset(
            combined_dataset,
            num_train_samples,
            num_val_samples,
            seq_len,
            reverse,
            use_residual,
            smooth_data,
        )

        # Merge dataset_info and dataset_info_split
        dataset_info_train = dataset_info.merge(
            dataset_info_split[
                [
                    "combined_dataset_index",
                    "train_time_steps",
                    "num_train_samples",
                    "train_seq_len",
                    "smooth_data",
                    "use_residual",
                ]
            ],
            on="combined_dataset_index",
            how="outer",
        )
        dataset_info_val = dataset_info.merge(
            dataset_info_split[
                [
                    "combined_dataset_index",
                    "val_time_steps",
                    "num_val_samples",
                    "val_seq_len",
                    "smooth_data",
                    "use_residual",
                ]
            ],
            on="combined_dataset_index",
            how="outer",
        )

        # Delete the combined dataset column after merging (not necessary anymore)
        dataset_info_train.drop(columns=["combined_dataset_index"], inplace=True)
        dataset_info_val.drop(columns=["combined_dataset_index"], inplace=True)

        # Save the datasets and information about them
        # => Train and val. datasets contain the same neurons, but with different time steps and other information
        dataset_info.to_csv(
            os.path.join(log_dir, "dataset", f"combined_dataset_info.csv"),
            index=True,
            header=True,
        )
        dataset_info_train.to_csv(
            os.path.join(log_dir, "dataset", f"train_dataset_info.csv"),
            index=True,
            header=True,
        )
        dataset_info_val.to_csv(
            os.path.join(log_dir, "dataset", f"val_dataset_info.csv"),
            index=True,
            header=True,
        )
        if save:
            torch.save(
                train_dataset, os.path.join(log_dir, "dataset", f"train_dataset.pt")
            )
            torch.save(val_dataset, os.path.join(log_dir, "dataset", f"val_dataset.pt"))
            with open(
                os.path.join(log_dir, "dataset", f"combined_dataset.pickle"), "wb"
            ) as f:
                pickle.dump(combined_dataset, f)

        return train_dataset, val_dataset


if __name__ == "__main__":
    config = OmegaConf.load("configs/submodule/dataset.yaml")
    print(OmegaConf.to_yaml(config), end="\n\n")
    dataset_train = get_datasets(config.dataset.train)
    dataset_predict = get_datasets(config.dataset.predict)
