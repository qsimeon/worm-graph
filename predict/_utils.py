from predict._pkg import *


def model_predict(
    model: torch.nn.Module,
    data: torch.Tensor,
    timesteps: int = 1,
    context_window: int = MAX_TOKEN_LEN,
    mask: Union[torch.Tensor, None] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make predictions for all neurons on a dataset with a trained model.

    This function uses a trained model to predict the future
    neural activity given sequences of past neural activity.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model used to make predictions.
    data : torch.Tensor
        Calcium data tensor with shape (1, max_timesteps, num_neurons) or
        (max_timesteps, num_neurons).
    timesteps : int, optional, default: 1000
        Number of future timesteps to predict.
    context_window : int, optional, default: 100
        Number of previous timesteps to regress on for prediction.
    mask : torch.Tensor or None, optional, default: None
        Output mask with shape (num_neurons,) to apply to the predictions.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]:
        A tuple containing the predictions and targets as torch tensors.
        All tensors are of the same shape as the input calcium data.
    """
    # verify input shape
    data = data.squeeze(0)  # (max_timesteps, num_neurons)
    assert data.ndim == 2, "Calcium data has incorrect shape!"

    # get desired targets and inputs to model
    targets = data[:, :]
    inputs = data[:-timesteps, :]

    # get predictions from model
    model.eval()
    predictions = model.generate(
        inputs,
        timesteps,
        context_window,
        mask,
    ).squeeze(0)

    # Returned sequences are all the same shape as the input `calcium_data`
    return predictions, targets
