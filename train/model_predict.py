from utils import DEVICE as device

def model_predict(single_worm_dataset, model):
  """
  Makes predictions for all neurons in the given
  worm dataset using a trained model.
  """
  model = model.to(device)
  calcium_data = single_worm_dataset['data']
  # model in/out
  input = calcium_data.squeeze().to(device)
  output = model(input)
  # tagets/preds
  targets = (input[1:] - input[:-1]).detach().cpu()
  predictions = output[:-1].detach().cpu()
  return targets, predictions