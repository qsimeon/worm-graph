import matplotlib.pyplot as plt


def plot_before_after_weights(before_weights, after_weights, W_name=''):
  """
  Plot side-by-side the pair of weights from 
  before and after training.
  """
  fig, axs = plt.subplots(1,2)
  axs[0].imshow(before_weights)
  axs[0].set_title('Initial weights '+W_name)
  axs[1].imshow(after_weights)
  axs[1].set_title('Trained weights '+W_name)
  plt.show()