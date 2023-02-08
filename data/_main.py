from _utils import *


@hydra.main(version_base=None, config_path=".", config_name="dataset")
def get_dataset(config):
    # load the dataset
    dataset_name = config.name
    all_worms_dataset = load_dataset(dataset_name)
    print("Chosen dataset:", dataset_name, end="\n\n")
    # pick one worm at random
    worm = np.random.choice(list(all_worms_dataset.keys()))
    single_worm_dataset = pick_worm(all_worms_dataset, worm)
    print("Picked:", worm, end="\n\n")
    print("Dataset keys:", list(single_worm_dataset.keys()), end="\n\n")
    return single_worm_dataset


if __name__ == "__main__":
    get_dataset()
    # # Load the connectome data
    # dataset = CElegansDataset()
    # connectome = dataset[0]
    # # Investigate the C. elegans connectome graph
    # print()
    # print("C. elegans connectome graph loaded successfully!")
    # print(
    #     "Attributes:",
    #     "\n",
    #     connectome.keys,
    #     "\n",
    #     f"Num. nodes {connectome.num_nodes}, Num. edges {connectome.num_edges}, "
    #     f"Num. node features {connectome.num_node_features}",
    #     end="\n",
    # )
    # print(f"\tHas isolated nodes: {connectome.has_isolated_nodes()}")
    # print(f"\tHas self-loops: {connectome.has_self_loops()}")
    # print(f"\tIs undirected: {connectome.is_undirected()}")
    # print(f"\tIs directed: {connectome.is_directed()}")
    # print()

    #  # load a recent dataset
    # Uzel2022 = load_Uzel2022()
    # # get data for one worm
    # single_worm_dataset = Uzel2022["worm1"]
    # num_neurons = single_worm_dataset["num_neurons"]
    # neuron_to_idx = single_worm_dataset["neuron_to_idx"]
    # calcium_data = single_worm_dataset["data"]
    # data = torch.nn.functional.pad(
    #     calcium_data, (0, 9), "constant", 0
    # )  # pad feature dimension to 10D
    # # create a dataset and data-loader
    # feature_mask = torch.tensor([1, 1] + 8 * [0]).to(
    #     torch.bool
    # )  # selects 2 features out of 10
    # dataset = MapDataset(data, feature_mask=feature_mask)
    # loader = torch.utils.data.DataLoader(
    #     dataset, batch_sampler=BatchSampler(dataset.batch_indices)
    # )
    # X, Y, meta = next(iter(loader))
    # # output properties of the dataset and data loader
    # print("size", dataset.size, "feature", dataset.num_features, end="\n\n")
    # print(
    #     X.shape,
    #     Y.shape,
    #     {k: meta[k][0] for k in meta},
    #     list(map(lambda x: x.shape, meta.values())),
    #     end="\n\n",
    # )
