from predict._pkg import *

# Init logger
logger = logging.getLogger(__name__)


def model_predict(
    log_dir: str,
    model: torch.nn.Module,
    experimental_datasets: dict,
    context_window: int,
):
    """
    Make predictions on a dataset with a trained model.

    Saves in log/predict a .csv with the Calcium neural activity predictions.

    Parameters:
    ----------
    log_dir : str
        Directory to save the predictions.
    model : torch.nn.Module
        Trained model.
    experimentad_datasets : dict
        A dictionary mapping the names of the experimental datasets to worms to predict.
    context_window : int
        Number of time steps to use as context for the predictions.
    """
    # Convert DictConfig to dict
    if isinstance(experimental_datasets, DictConfig):
        experimental_datasets = OmegaConf.to_object(experimental_datasets)

    # Retrieve information from training
    train_dataset_info = pd.read_csv(
        os.path.join(log_dir, "dataset", "train_dataset_info.csv"),
        converters={"neurons": ast.literal_eval},
    )
    seq_len = int(train_dataset_info["train_seq_len"].values[0])
    use_residual = int(train_dataset_info["use_residual"].values[0])
    smooth_data = int(train_dataset_info["smooth_data"].values[0])
    train_split_first = int(train_dataset_info["train_split_first"].values[0])
    train_split_ratio = float(train_dataset_info["train_split_ratio"].values[0])
    key_data = "residual_calcium" if use_residual else "calcium_data"
    key_data = "smooth_" + key_data if smooth_data else key_data

    # Load dataset with
    combined_dataset, dataset_info = create_combined_dataset(
        experimental_datasets=experimental_datasets,
        num_named_neurons=None,  # use all available neurons
    )

    # Put model on device
    model = model.to(DEVICE)

    # Iterate over combined datasets (same process as in split_combined_dataset in data/_utils.py)
    for _, single_worm_dataset in combined_dataset.items():
        # Extract relevant features from the dataset
        data = single_worm_dataset[key_data]
        neurons_mask = single_worm_dataset["named_neurons_mask"]
        worm_dataset = single_worm_dataset["dataset"]
        original_wormID = single_worm_dataset["original_worm"]

        # Query and save the named neurons to plot predictions afterwards
        neurons = dataset_info.query(
            'dataset == "{}" and original_index == "{}"'.format(worm_dataset, original_wormID)
        )["neurons"].iloc[0]
        # Now create the DataFrame
        neuron_df = pd.DataFrame({"named_neurons": neurons})

        # The index where to split the data
        split_idx = (
            int(train_split_ratio * len(data))
            if train_split_first
            else int((1 - train_split_ratio) * len(data))
        )
        assert (
            isinstance(seq_len, int) and 0 < seq_len < split_idx
        ), f"seq_len must be an integer > 0 and < {np.floor(train_split_ratio * len(data))}"

        # Split the data and the time vector into two sections
        data_splits = np.array_split(data, indices_or_sections=[split_idx], axis=0)

        # Separate the splits into training and validation sets
        if train_split_first:
            train_data_splits, val_data_splits = data_splits[::2], data_splits[1::2]
        else:
            train_data_splits, val_data_splits = data_splits[1::2], data_splits[::2]

        # Predictions using the first train split
        auto_reg_generated_activity_train = (
            model.generate(
                input=train_data_splits[0].unsqueeze(0).to(DEVICE),
                mask=neurons_mask.unsqueeze(0).to(DEVICE),
                num_new_timesteps=context_window,
                context_window=context_window,
            )
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )  # autoregressive generation
        # Create directories for saving results
        os.makedirs(
            os.path.join(log_dir, "prediction", "train", worm_dataset), exist_ok=True
        )  # dataset level
        os.makedirs(
            os.path.join(log_dir, "prediction", "train", worm_dataset, original_wormID),
            exist_ok=True,
        )  # worm level
        # Save results in dataframes
        result_df = prediction_dataframe_parser(
            x=train_data_splits[0],
            context_window=context_window,
            num_new_timesteps=context_window,
            auto_reg_generated_activity=auto_reg_generated_activity_train,
        )
        result_df.to_csv(
            os.path.join(
                log_dir,
                "prediction",
                "train",
                worm_dataset,
                original_wormID,
                "predictions.csv",
            )
        )
        neuron_df.to_csv(
            os.path.join(
                log_dir,
                "prediction",
                "train",
                worm_dataset,
                original_wormID,
                "named_neurons.csv",
            )
        )

        # Predictions using the first validation split
        auto_reg_generated_activity_val = (
            model.generate(
                input=val_data_splits[0].unsqueeze(0).to(DEVICE),
                mask=neurons_mask.unsqueeze(0).to(DEVICE),
                num_new_timesteps=context_window,
                context_window=context_window,
            )
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )  # autorregressive generation
        # Save the results
        os.makedirs(
            os.path.join(log_dir, "prediction", "val", worm_dataset), exist_ok=True
        )  # dataset level
        os.makedirs(
            os.path.join(log_dir, "prediction", "val", worm_dataset, original_wormID),
            exist_ok=True,
        )  # worm level
        # Save results in dataframes
        result_df = prediction_dataframe_parser(
            x=val_data_splits[0],
            context_window=context_window,
            num_new_timesteps=context_window,
            auto_reg_generated_activity=auto_reg_generated_activity_val,
        )
        result_df.to_csv(
            os.path.join(
                log_dir,
                "prediction",
                "val",
                worm_dataset,
                original_wormID,
                "predictions.csv",
            )
        )
        neuron_df.to_csv(
            os.path.join(
                log_dir,
                "prediction",
                "val",
                worm_dataset,
                original_wormID,
                "named_neurons.csv",
            )
        )


def prediction_dataframe_parser(
    x,
    context_window,
    num_new_timesteps,
    auto_reg_generated_activity,
):
    context_activity = x[: context_window + 1, :].detach().cpu().numpy()  # +1 for plot continuity
    ground_truth_activity = (
        x[context_window : context_window + num_new_timesteps, :].detach().cpu().numpy()
    )

    # Convert each tensor into a DataFrame and add type level
    df_context = pd.DataFrame(context_activity, columns=NEURONS_302)
    df_context["Type"] = "Context"
    df_context.set_index("Type", append=True, inplace=True)

    df_ground_truth = pd.DataFrame(ground_truth_activity, columns=NEURONS_302)
    df_ground_truth["Type"] = "Ground Truth"
    df_ground_truth.set_index("Type", append=True, inplace=True)

    df_ar_generated = pd.DataFrame(auto_reg_generated_activity, columns=NEURONS_302)
    df_ar_generated["Type"] = "AR Generation"
    df_ar_generated.set_index("Type", append=True, inplace=True)

    # Concatenate the DataFrames
    result_df = pd.concat([df_context, df_ground_truth, df_ar_generated])
    result_df = result_df.reorder_levels(["Type", None]).sort_index()

    return result_df
