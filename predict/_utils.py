from predict._pkg import *

# Init logger
logger = logging.getLogger(__name__)


def model_predict(
        log_dir: str,
        model: torch.nn.Module,
        experimental_datasets: DictConfig,
        context_window: int,
):
    """Make predictions on a dataset with a trained model.

    Saves in log/predict a .csv with the Calcium neural activity predictions.

    Parameters:
    ----------
    log_dir : str
        Directory to save the predictions.
    model : torch.nn.Module
        Trained model.
    experimentad_datasets : DictConfig
        Configuration file with name of the experimental datasets and worms to predict.
    dataset_type : str
        Type of dataset to predict. Either 'train' or 'val'.
    context_window : int
        Number of time steps to use as context for the predictions.
    nb_ts_to_generate : int
        Number of time steps to generate.
    """

    # Retrieve information from training
    train_dataset_info = pd.read_csv(os.path.join(log_dir, 'dataset', 'train_dataset_info.csv'))
    seq_len = int(train_dataset_info['train_seq_len'].values[0])
    k_splits = int(train_dataset_info['k_splits'].values[0])
    use_residual = int(train_dataset_info['use_residual'].values[0])
    smooth_data = int(train_dataset_info['smooth_data'].values[0])
    key_data = "residual_calcium" if use_residual else "calcium_data"
    key_data = "smooth_" + key_data if smooth_data else key_data

    assert context_window <= seq_len, (
        "The context window must be smaller than the sequence length (model was trained with seq_len = {}).".format(seq_len)
    )

    # Load dataset
    combined_dataset, dataset_info = create_combined_dataset(experimental_datasets=experimental_datasets,
                                                             num_named_neurons='all')

    model = model.to(DEVICE)

    # Iterate over combined datasets (same process as in split_combined_dataset in data/_utils.py)
    for wormID, single_worm_dataset in combined_dataset.items():

        # Extract relevant features from the dataset 
        data = single_worm_dataset[key_data]
        neurons_mask = single_worm_dataset["named_neurons_mask"]
        time_vec = single_worm_dataset["time_in_seconds"]
        worm_dataset = single_worm_dataset["dataset"]
        original_wormID = single_worm_dataset["original_worm"]
        
        # Split the data and the time vector into k splits
        data_splits = np.array_split(data, k_splits)
        time_vec_splits = np.array_split(time_vec, k_splits)

        # Separate the splits into training and validation sets
        train_data_splits = data_splits[::2]
        train_time_vec_splits = time_vec_splits[::2]
        val_data_splits = data_splits[1::2]
        val_time_vec_splits = time_vec_splits[1::2]

        # Predictions (GT and AR) using the first TRAIN split
        gt_generated_activity_train = model.generate(
            input=train_data_splits[0].unsqueeze(0).to(DEVICE),
            mask=neurons_mask.unsqueeze(0).to(DEVICE),
            nb_ts_to_generate=train_data_splits[0].shape[0]-context_window,
            context_window=context_window,
            autoregressive=False,
        ).squeeze(0).detach().cpu().numpy()

        auto_reg_generated_activity_train = model.generate(
            input=train_data_splits[0].unsqueeze(0).to(DEVICE),
            mask=neurons_mask.unsqueeze(0).to(DEVICE),
            nb_ts_to_generate=train_data_splits[0].shape[0]-context_window,
            context_window=context_window,
            autoregressive=True,
        ).squeeze(0).detach().cpu().numpy()

        # Save the results
        os.makedirs(os.path.join(log_dir, 'prediction', 'train', worm_dataset), exist_ok=True) # ds level
        os.makedirs(os.path.join(log_dir, 'prediction', 'train', worm_dataset, original_wormID), exist_ok=True) # worm level

        result_df = prediction_dataframe_parser(
            x = train_data_splits[0],
            context_window = context_window,
            gt_generated_activity = gt_generated_activity_train,
            auto_reg_generated_activity = auto_reg_generated_activity_train,
        )
        result_df.to_csv(os.path.join(log_dir, 'prediction', 'train', worm_dataset, original_wormID, "predictions.csv"))

        # Predictions (GT and AR) using the first VALIDATION split
        gt_generated_activity_val = model.generate(
            input=val_data_splits[0].unsqueeze(0).to(DEVICE),
            mask=neurons_mask.unsqueeze(0).to(DEVICE),
            nb_ts_to_generate=val_data_splits[0].shape[0]-context_window,
            context_window=context_window,
            autoregressive=False,
        ).squeeze(0).detach().cpu().numpy()

        auto_reg_generated_activity_val = model.generate(
            input=val_data_splits[0].unsqueeze(0).to(DEVICE),
            mask=neurons_mask.unsqueeze(0).to(DEVICE),
            nb_ts_to_generate=val_data_splits[0].shape[0]-context_window,
            context_window=context_window,
            autoregressive=True,
        ).squeeze(0).detach().cpu().numpy()

        # Save the results
        os.makedirs(os.path.join(log_dir, 'prediction', 'val', worm_dataset), exist_ok=True) # ds level
        os.makedirs(os.path.join(log_dir, 'prediction', 'val', worm_dataset, original_wormID), exist_ok=True) # worm level

        result_df = prediction_dataframe_parser(
            x = val_data_splits[0],
            context_window = context_window,
            gt_generated_activity = gt_generated_activity_val,
            auto_reg_generated_activity = auto_reg_generated_activity_val,
        )
        result_df.to_csv(os.path.join(log_dir, 'prediction', 'val', worm_dataset, original_wormID, "predictions.csv"))

        # Query and save the named neurons to plot predictions afterwards
        neurons = dataset_info.query('dataset == "{}" and original_index == "{}"'.format(worm_dataset, original_wormID))['neurons'].iloc[0]
        neuron_df = pd.DataFrame(neurons, columns=['named_neurons'])
        neuron_df.to_csv(os.path.join(log_dir, 'prediction', 'train', worm_dataset, original_wormID, "named_neurons.csv"))
        neuron_df.to_csv(os.path.join(log_dir, 'prediction', 'val', worm_dataset, original_wormID, "named_neurons.csv"))

def prediction_dataframe_parser(x, context_window, gt_generated_activity, auto_reg_generated_activity):
    context_activity = x[:context_window+1, :].detach().cpu().numpy() # +1 for plot continuity
    ground_truth_activity = x[context_window:, :].detach().cpu().numpy()

    # Convert each tensor into a DataFrame and add type level
    df_context = pd.DataFrame(context_activity, columns=NEURONS_302)
    df_context['Type'] = 'Context'
    df_context.set_index('Type', append=True, inplace=True)

    df_ground_truth = pd.DataFrame(ground_truth_activity, columns=NEURONS_302)
    df_ground_truth['Type'] = 'Ground Truth'
    df_ground_truth.set_index('Type', append=True, inplace=True)

    df_gt_generated = pd.DataFrame(gt_generated_activity, columns=NEURONS_302)
    df_gt_generated['Type'] = 'GT Generation'
    df_gt_generated.set_index('Type', append=True, inplace=True)

    df_ar_generated = pd.DataFrame(auto_reg_generated_activity, columns=NEURONS_302)
    df_ar_generated['Type'] = 'AR Generation'
    df_ar_generated.set_index('Type', append=True, inplace=True)

    # Concatenate the DataFrames
    result_df = pd.concat([df_context, df_ground_truth, df_gt_generated, df_ar_generated])
    result_df = result_df.reorder_levels(['Type', None]).sort_index()

    return result_df