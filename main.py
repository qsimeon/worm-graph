from pkg import *


@hydra.main(version_base=None, config_path="conf", config_name="main")
def pipeline(config: DictConfig) -> None:
    """
    Runs a complete pipeline using the parameters in main.yaml.
    Calls the below subroutines with parameters in their
    corresponding config files:
        process_data: preprocess.yaml
        get_dataset: dataset.yaml
        get_model: model.yaml
        train_model: train.yaml
        plot_figures: visualize.yaml
        analyze_outputs: analysis.yaml
    """
    # skips if data already preprocessed
    process_data(config)

    # returns a generator of single worm datasets
    dataset = get_dataset(config)

    model = get_model(config)

    model, log_dir = train_model(model, dataset, config, shuffle=True, optimizer=None)

    plot_figures(config, log_dir)
    # TODO: analysis
    return None


if __name__ == "__main__":
    pipeline()
