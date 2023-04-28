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
    # get input and output
    input = calcium_data.detach().to(DEVICE)
    # TODO: Why does this make such a big difference in prediction?
    # output = model(
    #     input.unsqueeze(1), tau,
    # ).squeeze(1)  # (max_timesteps, 1, NUM_NEURONS), batch_size = max_timesteps, seq_len = 1
    output = model(
        input.unsqueeze(0),
        tau=tau,
    ).squeeze(
        0
    )  # (1, max_timesteps, NUM_NEURONS),  batch_size = 1, seq_len = max_timesteps
    # targets and predictions
    targets = torch.nn.functional.pad(input.detach().cpu()[tau:], (0, 0, 0, tau))
    # prediction of the input shifted by tau
    predictions = output.detach().cpu()
    return targets, predictions
