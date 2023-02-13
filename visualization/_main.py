from visualization._utils import *


def plot_figures(config: DictConfig, log_dir: str = None) -> None:
    if log_dir is None:
        log_dir = config.visualize.default_log_dir
    # loss curves
    plot_loss_curves(log_dir)
    # calcium residuals
    # plot_targets_predictions(log_dir)

    # plot scatterplot of all neuron predictions
    # plot_correlation_scatterplot(log_dir)

    return None


if __name__ == "__main__":
    config = OmegaConf.load("conf/visualize.yaml")
    print("config:", OmegaConf.to_yaml(config), end="\n\n")
    plot_figures(
        config, log_dir=os.path.join("logs", "Uzel2022-LinearNN-2023_02_12_23_08_05")
    )
