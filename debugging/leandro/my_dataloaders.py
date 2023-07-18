import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from train._utils import split_train_test

class MaskedDataset(Dataset):
    def __init__(self, base_dataset, masks):
        self.base_dataset = base_dataset
        self.masks = masks

    def __getitem__(self, index):
        x, y, metadata = self.base_dataset[index]
        mask = self.masks[index]
        return x, y, metadata, mask

    def __len__(self):
        return len(self.base_dataset)

def get_calcium_data(worm_data, neuron_names=None, use_residual=False, use_mask=True):
    # Get one worm relevant data for training

    if use_residual:
        calcium_data = worm_data['smooth_residual_calcium']
    else:
        calcium_data = worm_data['smooth_calcium_data']

    time_vector = worm_data['time_in_seconds']

    # Verify if we have the neurons in the dataset
    if neuron_names is not None:
        for neuron in neuron_names:
            assert neuron in worm_data['named_neuron_to_slot'], f"We don't have data of neuron {neuron} for this worm"
        source_neurons_idx = [worm_data['named_neuron_to_slot'][neuron] for neuron in neuron_names] # Subset of labeled neurons
        mask = np.zeros(calcium_data.shape[1], dtype=bool)
        mask[source_neurons_idx] = True
    
    else:
        source_neurons_idx = [idx for idx in worm_data['named_neuron_to_slot'].values()] # Subset of labeled neurons
        mask = worm_data["named_neurons_mask"] # All named neurons

    if use_mask:
        return time_vector, calcium_data, mask
    else:
        return time_vector, calcium_data[:, source_neurons_idx], mask

def get_dataloaders(dict_dataset, number_cohorts, use_residual, k_splits, seq_len, num_samples,
                       reverse, tau, batch_size, shuffle_samples, desired_neurons_to_train, use_mask,
                       shuffle_worms):
    


    # === Train and test loaders ===

    # One cohort = one epoch
    # In one cohort we have X worms
    # We can extract Y time series samples from each worm
    # In each epoch we have sum_i=1^X Y_i samples

    # This is a way of keeping track of the number of worms that we need.

    # TODO: In order to chose the number of worms in a cohort, we can select the desired ones in the dict_dataset


    # === Parameters ===
    # use_residual = False
    # k_splits = 2 # Number of chunks to split the data: 1 chunk = 1 train/test split. Order: train, test, train, test, ...
    # seq_len = 100 # Number of time steps to extract from each chunk (time steps of each example)
    # num_samples = 100 # Total number of sample pairs (input,target) to extract from each worm
    # reverse = False # If True, the time series is reversed
    # tau = 10 # Number of time steps to shift the target (number of time steps we want to predict ahead)

    # batch_size = 32
    # shuffle_samples = False

    # desired_neurons_to_train = ['M3L']
    # use_mask = False # If use mask = True, the calcium data is (seq_len, 302). If false, the calcium data is (seq_len, desired_neurons_to_train)

    # shuffle_worms = True

    cohorts = sorted(dict_dataset.items()) * number_cohorts

    assert (len(cohorts) == number_cohorts * len(dict_dataset)), "Invalid number of worms."

    if shuffle_worms == True:
        cohorts = random.sample(cohorts, k=len(cohorts))

    # Split one cohort per epoch
    cohorts = np.array(np.array_split(cohorts, number_cohorts)) # Shape: (number_cohorts, number_worms, 2 - wormID and wormData)

    # === Creating the datasets ===

    # Memoize creation of data loaders and masks for speedup
    memo = {}

    cohort_trainloaders = np.empty(cohorts.shape[0], dtype=object)
    cohort_testloaders = np.empty(cohorts.shape[0], dtype=object)
    cohort_neuron_masks = np.empty(cohorts.shape[0], dtype=object) # We can set as we want

    for cohort_idx, cohort in enumerate(cohorts):
        # Train and test datasets for each worm
        train_datasets = np.empty(cohort.shape[0], dtype=object)
        if cohort_idx == 0:  # Keep the validation dataset the same
            test_datasets = np.empty(cohort.shape[0], dtype=object)
        neuron_masks = np.empty(cohort.shape[0], dtype=object)
        
        # Iterate over worms
        for worm_idx, (worm_id, worm_data) in enumerate(cohort):
            # If we have already loaded the worm...
            if worm_id in memo:
                train_datasets[worm_idx] = memo[worm_id]["train_dataset"] # Recover the train dataset of this worm
                if cohort_idx == 0:
                    test_datasets[worm_idx] = memo[worm_id]["test_dataset"] # Recover the test dataset of this worm
            
            # If we have not loaded the worm...
            else:

                time_vec, calcium_data, mask = get_calcium_data(worm_data, neuron_names=desired_neurons_to_train, use_mask=use_mask)

                train_datasets[worm_idx], test_dataset_tmp, _, _  = split_train_test(
                    data = calcium_data,
                    k_splits = k_splits,
                    seq_len = seq_len,
                    num_samples = num_samples,
                    time_vec = time_vec,
                    reverse = reverse,
                    tau = tau,
                    use_residual = use_residual,
                )
                if cohort_idx == 0:  # Keep the validation dataset the same
                        test_datasets[worm_idx] = test_dataset_tmp

                # Add to memo
                memo[worm_id] = dict(
                    train_dataset=train_datasets[worm_idx],
                    test_dataset=test_dataset_tmp,
                )

            neuron_masks[worm_idx] = mask

        # Here we apply the MaskedDataset to the concatenated datasets
        cohort_train_dataset = MaskedDataset(ConcatDataset(list(train_datasets)), [item for item in neuron_masks for _ in range(num_samples)])
        cohort_test_dataset = MaskedDataset(ConcatDataset(list(test_datasets)), [item for item in neuron_masks for _ in range(num_samples)])

        cohort_trainloaders[cohort_idx] = DataLoader(
            cohort_train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_samples,
            pin_memory=True,
            num_workers=0,
        )  # returns (X, Y, Dict, Mask) when iterating over it

        cohort_testloaders[cohort_idx] = DataLoader(
            cohort_test_dataset,
            batch_size=batch_size,
            shuffle=shuffle_samples,
            pin_memory=True,
            num_workers=0,
        )  # returns (X, Y, Dict, Mask) when iterating over it
        
    # Print number of examples in each cohort
    print(f"Cohort: {len(cohort)} worms")
    print(f"Train dataset: {len(cohort_trainloaders[cohort_idx])*batch_size} samples")
    print(f"Test dataset: {len(cohort_testloaders[cohort_idx])*batch_size} samples")

    return cohort_trainloaders, cohort_testloaders