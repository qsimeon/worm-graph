import matplotlib.pyplot as plt


def plot_correlation_scatter(targets, predictions, plt_title=""):
    """
    Create a scatterpot of the target and predicted residuals.
    """
    max_time = len(targets)
    xx_tr = targets[: max_time // 2, :]
    yy_tr = predictions[: max_time // 2, :]
    xx_te = targets[max_time // 2 :, :]
    yy_te = predictions[max_time // 2 :, :]
    # print model test results
    print()
    print("model test performance:", ((yy_te - xx_te) ** 2).mean())
    print()
    print("signs flipped:", ((-1 * yy_te - xx_te) ** 2).mean())
    print()
    print("baseline:", ((0 * yy_te - xx_te) ** 2).mean())
    # plot figures
    fig, axs = plt.subplots(1, 1)
    axs.scatter(xx_tr, yy_tr, c="m", alpha=0.7, label="train")
    axs.scatter(xx_te, yy_te, c="c", alpha=0.2, label="test")
    axs.axis("equal")
    axs.set_title(plt_title)
    axs.set_xlim([-1, 1])
    axs.set_ylim([-1, 1])
    axs.set_xlabel(r"True residual $\Delta F / F$")
    axs.set_ylabel(r"Predicted residual $\Delta F / F$")
    axs.legend()
    plt.show()
