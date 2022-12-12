import matplotlib.pyplot as plt

def plot_correlation_scatter(targets, predictions):
  """
  Create a scatterpot of the target and predicted residuals.
  """
  max_time = len(targets)
  xx_tr = targets[:max_time//2, :]
  yy_tr = predictions[:max_time//2, :]
  xx_te = targets[max_time//2:, :]
  yy_te = predictions[max_time//2:, :]

  print()
  print('model performance:', ((yy_tr - xx_tr)**2).mean())
  print()
  print('signs flipped:', ((-1*yy_tr - xx_tr)**2).mean())
  print()
  print('baseline:', ((0*yy_tr - xx_tr)**2).mean())

  fig, axs = plt.subplots(1,1)
  axs.scatter(xx_tr, yy_tr, c='m', alpha=0.7, label='train')
  axs.scatter(xx_te, yy_te, c='c', alpha=0.2, label='test')
  axs.axis('equal')
  axs.set_title('Linear model: correlation of neuron Ca2+ residuals')
  axs.set_xlabel(r"true $\Delta F / F$")
  axs.set_ylabel(r"predicted $\Delta F / F$")
  axs.legend()
  plt.show()