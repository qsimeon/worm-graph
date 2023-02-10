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
    # access the Hydra config:
    print(HydraConfig.get().job, end="\n\n")

    # skips if data already preprocessed
    process_data(config)

    # returns a generator  of single worm datatsets
    dataset = get_dataset(config)

    model = get_model(config)

    model, logs = train_model(model, dataset, config)

    # plot_figures(logs)
    # analysis
    return None


if __name__ == "__main__":
    pipeline()
