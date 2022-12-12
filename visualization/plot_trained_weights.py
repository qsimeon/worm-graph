import matplotlib.pyplot as plt

def plot_linear_weights(untrained_weights, trained_weights):
  """
  Plot the linear layer weights at initialization and after
  being trained side by side.
  """
  fig, axs = plt.subplots(1,2)
  axs[0].imshow(untrained_weights)
  axs[0].set_title('Initial weights')
  axs[1].imshow(trained_weights)
  axs[1].set_title('Trained weights')
  plt.show()