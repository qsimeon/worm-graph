from predict._pkg import *


def model_predict(
    model: torch.nn.Module,
    calcium_data: torch.Tensor,
    tau: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make predictions for all neurons on a dataset with a trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model used to make predictions.
    calcium_data : torch.Tensor
        Calcium data tensor with shape (1, max_timesteps, num_neurons) or
        (max_timesteps, num_neurons).
    tau : int, optional, default: 1
        Time-shift parameter for the predictions.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]:
        A tuple containing the padded inputs, predictions, and targets as torch tensors.
        All tensors are of the same shape as the input calcium data.
    """


    num_neurons = calcium_data.size(1)
    model = model.to(DEVICE)
    model.eval()

    # Model in/out
    calcium_data = calcium_data.squeeze(0) # (max_timesteps, num_neurons)
    assert (
        calcium_data.ndim == 2 and calcium_data.size(1) >= num_neurons
    ), "Calcium data has incorrect shape!"

    # Get input and output
    input = calcium_data.to(DEVICE)[:-tau, :] #? Why [:-tau, :] and not -1 directly?
    with torch.no_grad():
        output = model(
            input.unsqueeze(0),
            tau=tau,
        ).squeeze(
            0
        )  # (1, max_timesteps, num_neurons),  batch_size = 1, seq_len = max_timesteps

    # Get desired target
    target = calcium_data[tau:, :] #? Same question again

    # Inputs, predictions, and targets
    inputs = torch.nn.functional.pad(input.detach().cpu(), (0, 0, 0, tau))
    predictions = torch.nn.functional.pad(output.detach().cpu(), (0, 0, tau, 0))
    targets = torch.nn.functional.pad(target.detach().cpu(), (0, 0, tau, 0))

    # Returned sequences are all the same shape as the input `calcium_data`
    return inputs, predictions, targets
