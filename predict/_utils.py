from predict._pkg import *


def model_predict(
    model: torch.nn.Module,
    calcium_data: torch.Tensor,
    tau: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Makes predictions for all neurons in the
    calcium data tensor using a trained model.
    """
    NUM_NEURONS = calcium_data.size(1)
    # model = model.double().to(DEVICE)
    model = model.to(DEVICE)
    model.eval()
    # model in/out
    calcium_data = calcium_data.squeeze(0)
    assert (
        calcium_data.ndim == 2 and calcium_data.size(1) >= NUM_NEURONS
    ), "Calcium data has incorrect shape!"
    # get target
    target = calcium_data[tau:, :]
    # get input and output
    input = calcium_data.to(DEVICE)[:-tau, :]
    with torch.no_grad():
        output = model(  # CORRECT approach
            input.unsqueeze(0),
            tau=tau,
        ).squeeze(
            0
        )  # (1, max_timesteps, NUM_NEURONS),  batch_size = 1, seq_len = max_timesteps
    # inputs, predictions, and targets
    inputs = torch.nn.functional.pad(input.detach().cpu(), (0, 0, 0, tau))
    predictions = torch.nn.functional.pad(output.detach().cpu(), (0, 0, tau, 0))
    targets = torch.nn.functional.pad(target.detach().cpu(), (0, 0, tau, 0))
    # returned sequences are all the same shape as the input `calcium_data`
    return inputs, predictions, targets
