import numpy as np
import matplotlib.pyplot as plt

def plot_more_data_losses(results, plt_title=''):
  """
  Makes a plot of the aggregated loss curves on multiple dataset
  sizes from the `results` returned by `more_data_training`. 
  """
  num_datasets = len(results)
  fig, ax = plt.subplots(1,1)

  cmap = plt.get_cmap("PuRd", num_datasets+2)
  for i, res in enumerate(results):
    label = None if i<num_datasets-1 else 'train'
    model, log = res
    ax.plot(log['epochs'], np.log10(log['train_losses']), color=cmap(i), linewidth=2, label=label)

  cmap = plt.get_cmap("YlGnBu", num_datasets+2)
  for i, res in enumerate(results):
    label = None if i<num_datasets-1 else 'test'
    model, log = res
    ax.plot(log['epochs'], np.log10(log['test_losses']), color=cmap(i), linewidth=2, label=label)

  avg_base_train_loss = 0
  for i, res in enumerate(results):
    model, log = res
    avg_base_train_loss += np.log10(log['base_train_losses'])
  avg_base_train_loss /= i+1
  ax.plot(log['epochs'], avg_base_train_loss, color='purple', linewidth=4, linestyle='--', alpha=0.6, label='avg. train baseline')

  avg_base_test_loss = 0
  for i, res in enumerate(results):
    model, log = res
    avg_base_test_loss += np.log10(log['base_test_losses'])
  avg_base_test_loss /= i+1
  ax.plot(log['epochs'], avg_base_test_loss, color='green', linewidth=3, linestyle='-.', alpha=1, label='avg. test baseline')

  ax.legend(title='%s dataset sizes'%num_datasets, labelspacing=0, loc="lower left")
  ax.set_xlabel('Epoch')
  ax.set_ylabel('log MSE')
  ax.set_title(plt_title)
  plt.show()