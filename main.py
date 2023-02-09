from pkg import *


@hydra.main(version_base=None, config_path="conf", config_name="main")
def pipeline(config: DictConfig) -> None:
    """
    Runs a complete pipeline using params in main.yaml.
    Sub-functions and corresponding config files
        process_data: preprocess.yaml
        get_dataset: dataset.yaml
        get_model: model.yaml
        train_model: train.yaml
        analyze_outputs: analysis.yaml
    """
    print(OmegaConf.to_yaml(config), end="\n\n")

    process_data(config)  # skip if already preprocessed

    # `dataset` is a generator
    dataset = get_dataset(config)

    model = get_model(config)

    model, logs = train_model(config)

    # plot_figures(logs)
    return None


if __name__ == "__main__":
    pipeline()
