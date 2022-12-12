import matplotlib.pyplot as plt

def plot_target_prediction(target, prediction, nid):
    """
    Make a plot of predictions versus targets for a full dataset.
    nid: ID or name of neuron.
    """
    plt.figure()
    plt.plot(target, linestyle='-', label='target', linewidth=2)
    plt.plot(prediction, linestyle=':', label='prediction', linewidth=3)
    plt.axvline(x=len(target)//2, c='r', linestyle='--', linewidth=4)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Residual $\Delta F/F$')
    plt.title('Linear model Ca2+ residuals prediction on neuron %s'%nid)
    plt.show()