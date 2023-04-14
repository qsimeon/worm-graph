from utils import *


@hydra.main(version_base=None, config_path="conf", config_name="main")
def pipeline(
    config: DictConfig,
) -> None:
    """
    Runs a complete pipeline using the parameters in main.yaml.
    Calls the below subroutines with parameters in their
    corresponding config files:
        process_data: preprocess.yaml
        get_dataset: dataset.yaml
        get_model: model.yaml
        train_model: train.yaml
        plot_figures: visualize.yaml
        TODO: analyze_outputs: analysis.yaml
    """
    # print Pytorch device
    print("\ntorch device: %s" % (DEVICE), end="\n\n")

    # intialize random seeds
    np.random.seed(config.globals.random_seed)
    torch.manual_seed(config.globals.random_seed)
    random.seed(config.globals.random_seed)

    # skips if data already preprocessed
    process_data(config)

    # returns a generator of single worm datasets
    dataset = get_dataset(config)

    # get the model to train
    model = get_model(config)

    # train model is the bulk of the pipeline code
    model, log_dir = train_model(
        model,
        dataset,
        config,
        shuffle=config.globals.shuffle,  # shuffle worms
        log_dir=None,  # hydra changes working directory to log directory
    )

    # plot figures
    plot_figures(config, log_dir)
    ## TODO: analysis
    # analyze_outputs(config, log_dir)
    return None


if __name__ == "__main__":
    pipeline()
