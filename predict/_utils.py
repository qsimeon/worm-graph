from predict._pkg import *

# Init logger
logger = logging.getLogger(__name__)


def model_predict(
        log_dir: str,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        context_window: int,
        nb_ts_to_generate: int = None,
        worms_to_predict: list = None,
):
    
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

        # Skip example if from the same worm
        if metadata["wormID"] in worms_predicted:
            continue

        # Skip example if not in the list of worms to predict
        if worms_to_predict is not None and metadata["wormID"] not in worms_to_predict:
            continue

        # Send tensors to device
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        mask = mask.to(DEVICE)

        worms_predicted.add(metadata["wormID"])

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

        # Save the DataFrame
        result_df.to_csv(os.path.join(log_dir, f"{metadata['wormID']}.csv"))

    logger.info("Done. {}".format(worms_predicted))