from pkg import *


@hydra.main(version_base=None, config_path="conf", config_name="main")
def pipeline(config: DictConfig) -> None:
    """
    Runs a complete pipeline using params in main.yaml.
    Subroutines and corresponding their config files:
        process_data: preprocess.yaml
        get_dataset: dataset.yaml
        get_model: model.yaml
        train_model: train.yaml
        analyze_outputs: analysis.yaml
    """
    # skips if data already preprocessed
    process_data(config)

    # # returns a generator of single worm datasets
    # dataset = get_dataset(config)

    # model = get_model(config)

    # model, log_dir = train_model(model, dataset, config, shuffle=True, optimizer=None)

    # # plot_figures(log_dir)
    # # analysis
    return None


if __name__ == "__main__":
    pipeline()
