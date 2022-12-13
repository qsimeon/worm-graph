import matplotlib.pyplot as plt
import numpy as np


def plot_multi_worm_losses(results, plt_title=''):
  """
  Makes a plot of the aggregated loss curves on multiple worms
  from the `results` returned by `multi_worm_training`. 
  """
  num_worms = len(results)
  fig, ax = plt.subplots(1,1)

  cmap = plt.get_cmap("Reds", num_worms+2)
  for i, res in enumerate(results):
    label = None if i<num_worms-1 else 'train'
    model, log = res
    ax.plot(log['epochs'], np.log10(log['train_losses']), color=cmap(i+2), linewidth=2, label=label)

  cmap = plt.get_cmap("Blues", num_worms+2)
  for i, res in enumerate(results):
    label = None if i<num_worms-1 else 'test'
    model, log = res
    ax.plot(log['epochs'], np.log10(log['test_losses']), color=cmap(i+2), linewidth=2, label=label)

  avg_base_train_loss = 0
  for i, res in enumerate(results):
    model, log = res
    avg_base_train_loss += np.log10(log['base_train_losses'])
  avg_base_train_loss /= i+1
  ax.plot(log['epochs'], avg_base_train_loss, color='r', linewidth=4, linestyle='--', alpha=0.6, label='avg. train baseline')

  avg_base_test_loss = 0
  for i, res in enumerate(results):
    model, log = res
    avg_base_test_loss += np.log10(log['base_test_losses'])
  avg_base_test_loss /= i+1
  ax.plot(log['epochs'], avg_base_test_loss, color='b', linewidth=3, linestyle='-.', alpha=1, label='avg. test baseline')

  ax.legend(title='%s worms'%num_worms, labelspacing=0)
  ax.set_xlabel('Epoch')
  ax.set_ylabel('log MSE')
  ax.set_title(plt_title)
  plt.show()