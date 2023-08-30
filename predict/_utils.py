from predict._pkg import *

# Init logger
logger = logging.getLogger(__name__)


def model_predict(
        log_dir: str,
        model: torch.nn.Module,
        experimental_datasets: DictConfig,
        dataset_type: str,
        context_window: int,
        nb_ts_to_generate: int = None,
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
    num_train_samples = int(train_dataset_info['num_train_samples'].values[0])
    k_splits = int(train_dataset_info['k_splits'].values[0])
    tau = int(train_dataset_info['tau'].values[0])
    use_residual = int(train_dataset_info['use_residual'].values[0])
    smooth_data = int(train_dataset_info['smooth_data'].values[0])

    # Load dataset
    combined_dataset, _ = create_combined_dataset(experimental_datasets=experimental_datasets,
                                                    num_named_neurons='all')
    train_dataset, val_dataset, _ = split_combined_dataset(combined_dataset=combined_dataset,
                                                            k_splits=k_splits,
                                                            num_train_samples=num_train_samples,
                                                            num_val_samples=num_train_samples, # use the same number of samples as in the train dataset
                                                            seq_len=seq_len,
                                                            tau=tau,
                                                            use_residual=use_residual,
                                                            smooth_data=smooth_data,
                                                            reverse=False
                                                            )
    
    # Select desired dataset and prepare the filters for predictions
    dataset = train_dataset if dataset_type == 'train' else val_dataset
    datasets_to_predict = [dataset_name for dataset_name in experimental_datasets.keys() if experimental_datasets[dataset_name] is not None]

    x, y, mask, metadata = next(iter(dataset))
    seq_len = x.shape[0]

    assert context_window < seq_len, (
        "The context window must be smaller than the sequence length ({}).".format(seq_len)
    )

    nb_gt_ts_to_generate = seq_len - context_window # Time steps for ground truth comparison

    if nb_ts_to_generate is None:
        nb_ts_to_generate = nb_gt_ts_to_generate

    worms_predicted = set()

    model = model.to(DEVICE)

    # Iterate over the examples in the dataset
    for x, y, mask, metadata in iter(dataset):

        # Skip example if not in the list of datasets to predict
        if metadata["worm_dataset"] not in datasets_to_predict:
            continue

        # Filter worms to predict
        worms_to_predict = experimental_datasets[metadata["worm_dataset"]]

        assert not isinstance(worms_to_predict, int), (
            "The worms you want to predict must be a string or a list of strings."
        )

        if isinstance(worms_to_predict, str):
            if worms_to_predict == 'all':
                worms_to_predict = [metadata["wormID"]]
            else:
                worms_to_predict = [worms_to_predict]
        
        # Skip example if already predicted
        if (metadata["wormID"], metadata["worm_dataset"]) in worms_predicted:
            continue


        # Send tensors to device
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        mask = mask.to(DEVICE)

        worms_predicted.add((metadata["wormID"], metadata["worm_dataset"]))

        gt_generated_activity = model.generate(
            input=x.unsqueeze(0),
            mask=mask.unsqueeze(0),
            nb_ts_to_generate=nb_gt_ts_to_generate,
            context_window=context_window,
            autoregressive=False,
        ).squeeze(0).detach().cpu().numpy()

        auto_reg_generated_activity = model.generate(
            input=x.unsqueeze(0),
            mask=mask.unsqueeze(0),
            nb_ts_to_generate=nb_ts_to_generate,
            context_window=context_window,
            autoregressive=True,
        ).squeeze(0).detach().cpu().numpy()
        
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

        # Prediction folder
        pred_dir = os.path.join(log_dir, 'prediction', dataset_type)

        # Create folder for dataset
        os.makedirs(os.path.join(pred_dir, metadata['worm_dataset']), exist_ok=True) # ds level
        os.makedirs(os.path.join(pred_dir, metadata['worm_dataset'], metadata['wormID']), exist_ok=True) # worm level
        # Save the DataFrame
        result_df.to_csv(os.path.join(pred_dir, metadata['worm_dataset'], metadata['wormID'], "predictions.csv"))

    logger.info("Done. {}".format(worms_predicted))