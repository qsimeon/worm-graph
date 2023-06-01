import mat73
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from preprocess._utils import DiffTVR
from utils import ROOT_DIR, RAW_FILES, NEURONS_302, VALID_DATASETS, MATLAB_FILES
import torch
import logging
from scipy.signal import savgol_filter
from scipy.linalg import solve
from typing import Tuple, Union
import pickle
from testing.leandro.filters import *

processed_path = os.path.join(ROOT_DIR, "data/processed/neural")

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log message format
    datefmt='%d-%b-%y %H:%M:%S',  # Define the date/time format
    filename= ROOT_DIR+"/logs/execution/preprocess_notebook.log",  # Specify the log file (optional)
    filemode='w'  # Set the file mode (optional, default is 'a' for appending)
)

class PreprocessDataset:
    """Preprocesses the data for a given dataset."""

    def __init__(self, dataset, transform, smooth_method, resample_dt, norm_dim='neurons', pckl=True):
        # Saving the arguments
        self.dataset = dataset
        self.transform = transform
        self.resample_dt = resample_dt

        assert smooth_method.lower() in ['fft', 'sg', 'tvr'], \
            "Invalid smooth_method! Please pick one from: ['fft', 'sg', 'tvr']"
        self.smooth_method = smooth_method.lower()

        assert norm_dim.lower() in ['neurons', 'time'], \
            "Invalid norm_dim! Please pick one from: ['neurons', 'time']" 
        self.norm_dim = norm_dim.lower()

        self.pckl = pckl # if True, saves all the intemeriary data
        self.all_raw_data = {}
        self.data = {}

    def get_files_and_features(self):
        # Verify if the dataset is valid
        assert (self.dataset in VALID_DATASETS
        ), "Invalid dataset requested! Please pick one from:\n{}".format(
            list(VALID_DATASETS)
        )

        matfiles = MATLAB_FILES[self.dataset][0]
        features = MATLAB_FILES[self.dataset][1]

        return matfiles, features
    
    def load_mat(self):
        """Load all the .mat files of a given dataset.

        Parameters
        ----------
        dataset : str
            Name of the dataset to be loaded.
            Options are {Kato2015, Nichols2017, Skora2018, Kaplan2020,
                         Uzel2022}
        matlabfiles : list
            List containing the names of the .mat files to be loaded.
            Don't include the .mat extension in the names.
        features : list
            List containing the dictionaries with the arguments corresponding
            to each .mat file.

        Returns
        -------
        data : dict
            Dictionary containing the loaded data.
        """

        # Get the files and features for the dataset
        matfiles, features = self.get_files_and_features()

        # Load the data and save in a dict. The keys are the .mat files names

        for i, matf in enumerate(matfiles):
            
            ft = features[i] # Get the features for this file

            raw_data = mat73.loadmat(os.path.join(ROOT_DIR, 'opensource_data', self.dataset, matf+'.mat'))[matf] # Get raw data (dict)
            #logging.info("{} - Raw keys ({}) = {}".format(self.dataset+'/'+matf, matf, list(raw_data.keys())))

            # Extract relevant data (all worms)
            all_IDs = raw_data[ft['ids']]  # Identified neuron IDs (only subset have neuron names) # (num_worms, id_len:vary)
            all_traces = raw_data[ft['traces']]  # Neural activity traces corrected for bleaching (worms) # (num_worms, traces:vary)
            timeVectorSeconds = raw_data[ft['tv']] # (num_worms, traces:vary)
            #logging.info("{} - Num. worms ({}) = {}".format(self.dataset+'/'+matf, matf, len(all_IDs)))

            self.all_raw_data[matf] = {
                'ids': all_IDs,
                'traces': all_traces,
                'tvs': timeVectorSeconds
            }

            logging.info("{}/{} loaded.".format(self.dataset, matf))

    def _pick_non_none(self, l):
        """Returns the first non-None element in a list (l).
        """
        for i in range(len(l)):
            if l[i] is not None:
                return l[i]

    def find_unique(self, ids, calcium_data):
        """Makes a mapping between the neuron IDs and their indices.

        Parameters
        ----------
        ids : list
            List containing the neuron IDs of a single worm.

        Returns
        -------
        idx : list
            List containing the indices of the unique neurons of this worm.
        neuron_to_idx : dict
            Dictionary mapping the neuron IDs to their indices.
        """

        # Pre-processing the neuron IDs
        ids = [
            (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
            for _, j in enumerate(ids)
        ] # Position on the list or name of the neuron

        _, idx = np.unique(
            ids, return_index=True
        ) # Get unique neurons and their indices

        ids = [ids[_] for _ in idx]  # Only keep unique neuron IDs
        calcium_data = calcium_data[:, idx.astype(int)]  # Only get data for unique neurons

        # Mapping unique neuron IDs to indices
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(ids)
        } # Unlabeled neurons are given their index as name, labeled neurons are given their name

        # Format the neuron names if it finishes with 0
        neuron_to_idx = {
            nid: (
                name.replace("0", "")
                if not name.endswith("0") and not name.isnumeric()
                else name
            )
            for nid, name in neuron_to_idx.items()
        }

        # Invert the mapping
        neuron_to_idx = dict(
            (v, k) for k, v in neuron_to_idx.items()
        )

        return ids, calcium_data, neuron_to_idx
    
    def apply_transform(self, calcium_data):
        """Applies the transform to the calcium data.

        Parameters
        ----------
        calcium_data : np.ndarray
            Calcium data to be transformed.

        Returns
        -------
        calcium_data : np.ndarray
            Transformed calcium data.
        """

        if self.norm_dim == 'neurons':
            calcium_data = self.transform.fit_transform(calcium_data)
            calcium_data = torch.tensor(calcium_data, dtype=torch.float32)
        elif self.norm_dim == 'time':
            calcium_data = self.transform.fit_transform(calcium_data.T).T
            calcium_data = torch.tensor(calcium_data, dtype=torch.float32)

        return calcium_data
    
    def interpolate_data(self, time, data):
        """Interpolate data using np.interp, with support for torch.Tensor.

        This function takes the given time points and corresponding data and
        interpolates them to create new data points with the desired time 
        interval. The input tensors are first converted to NumPy arrays for 
        interpolation, and the interpolated data and time points are then 
        converted back to torch.Tensor objects before being returned.

        Parameters
        ----------
        time : torch.Tensor
            1D tensor containing the time points corresponding to the data.
        data : torch.Tensor
            A 2D tensor containing the data to be interpolated, with shape
            (time, neurons).
        target_dt : float
            The desired time interval between the interpolated data points.
            If None, no interpolation is performed.

        Returns
        -------
        torch.Tensor, torch.Tensor:
            Two tensors containing the interpolated time points and data.
        """

        # If target_dt is None, return the original data
        if self.resample_dt is None:
            return time, data

        # Convert input tensors to NumPy arrays
        time_np = time.squeeze().numpy()
        data_np = data.numpy()

        # Interpolate the data
        target_time_np = np.arange(time_np.min(), time_np.max(), self.resample_dt)
        num_neurons = data_np.shape[1]
        interpolated_data_np = np.zeros((len(target_time_np), num_neurons))

        for i in range(num_neurons):
            interpolated_data_np[:, i] = np.interp(target_time_np, time_np, data_np[:, i])

        # Convert the interpolated data and time back to torch.Tensor objects
        target_time = torch.from_numpy(target_time_np).to(torch.float32).unsqueeze(-1)
        interpolated_data = torch.from_numpy(interpolated_data_np).to(torch.float32)

        return target_time, interpolated_data

    def smooth_data_preprocess(self, calcium_data):
        """Smooth the calcium data. Returns the denoised signals calcium signals using FFT.

        Parameters
        ----------
            calcium_data: tensor
                Raw calcium imaging data to smooth

        Returns
        -------
            smooth_ca_data: tensor
                Smoothed calcium data
            residual: tensor
                Residual from the original data (calcium_data)
            residual_smooth_ca_data: tensor
                Residual from the smoothed data (smooth_ca_data)
        """

        # Number of time steps
        n = calcium_data.shape[0]

        # Initialize the size for smooth_calcium_data
        smooth_ca_data = torch.zeros_like(calcium_data)

        # Calculate original residual
        residual = torch.zeros_like(calcium_data)
        residual[1:] = calcium_data[1:] - calcium_data[: n - 1]

        # Savitzky-Golay method
        if str(self.smooth_method).lower() == "sg" or self.smooth_method == None:
            smooth_ca_data = savgol_filter(calcium_data, 5, 3, mode="nearest", axis=-1)

        # FFT method
        elif str(self.smooth_method).lower() == "fft":
            data_torch = calcium_data
            smooth_ca_data = torch.zeros_like(calcium_data)
            max_timesteps, num_neurons = data_torch.shape
            frequencies = torch.fft.rfftfreq(max_timesteps, d=self.resample_dt)  # dt: sampling time
            threshold = torch.abs(frequencies)[int(frequencies.shape[0] * 0.1)]
            oneD_kernel = torch.abs(frequencies) < threshold
            fft_input = torch.fft.rfftn(data_torch, dim=0)
            oneD_kernel = oneD_kernel.repeat(calcium_data.shape[1], 1).T
            fft_result = torch.fft.irfftn(fft_input * oneD_kernel, dim=0)
            smooth_ca_data[0 : min(fft_result.shape[0], calcium_data.shape[0])] = fft_result

        # TVR method
        elif str(self.smooth_method).lower() == "tvr":
            diff_tvr = DiffTVR(n, 1)
            for i in range(0, calcium_data.shape[1]):
                temp = np.array(calcium_data[:, i])
                temp.reshape(len(temp), 1)
                (item_denoise, _) = diff_tvr.get_deriv_tvr(
                    data=temp,
                    deriv_guess=np.full(n + 1, 0.0),
                    alpha=0.005,
                    no_opt_steps=100,
                )
                smooth_ca_data[:, i] = torch.tensor(item_denoise[: (len(item_denoise) - 1)])

        m = smooth_ca_data.shape[0]
        residual_smooth_ca_data = torch.zeros_like(residual)
        residual_smooth_ca_data[1:] = smooth_ca_data[1:] - smooth_ca_data[: m - 1]

        return smooth_ca_data, residual, residual_smooth_ca_data

    def update_non_std_dict(self, data_dict, worm, calcium_data, smooth_calcium_data,
                            residual, smooth_residual, neuron_to_idx,
                            time_in_seconds, num_named):
        data_dict.update(
                    {
                        worm: {
                            "dataset": self.dataset,
                            "smooth_method": self.smooth_method.upper(),
                            # "worm": worm,
                            "calcium_data": calcium_data,
                            "smooth_calcium_data": smooth_calcium_data,
                            "residual_calcium": residual,
                            "smooth_residual_calcium": smooth_residual,
                            "neuron_to_idx": neuron_to_idx,
                            "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                            "max_timesteps": calcium_data.shape[0],
                            "time_in_seconds": time_in_seconds,
                            "dt": self.resample_dt,
                            "num_neurons": calcium_data.shape[1],
                            "num_named_neurons": num_named,
                            "num_unknown_neurons": calcium_data.shape[1] - num_named,
                        },
                    }
                )

    def reshape_calcium_data(self, single_worm_dataset):
        """Standardizes the calcium data to a (max_timesteps, 302) shape.
        
        Inserts neuron masks and mappings of neuron labels to indices in the data.

        Parameters
        ----------
        single_worm_dataset : dict
            Dictionary containing the calcium data for a single worm.
        
        Returns
        -------
        single_worm_dataset : dict
            Dictionary containing the reshaped dataset.
        """

        # Get the calcium data for this worm
        origin_calcium_data = single_worm_dataset["calcium_data"]
        smooth_calcium_data = single_worm_dataset["smooth_calcium_data"]
        residual_calcium = single_worm_dataset["residual_calcium"]
        smooth_residual_calcium = single_worm_dataset["smooth_residual_calcium"]

        # Get the number of unidentified tracked neurons
        num_unknown_neurons = single_worm_dataset["num_unknown_neurons"]

        # Get the neuron to idx map of this worm
        neuron_to_idx = single_worm_dataset["neuron_to_idx"]
        idx_to_neuron = single_worm_dataset["idx_to_neuron"]

        # Get the length of the time series
        max_timesteps = single_worm_dataset["max_timesteps"]

        # Load names of all 302 neurons
        neurons_302 = NEURONS_302

        # Check the calcium data
        assert len(idx_to_neuron) == origin_calcium_data.size(
            1
        ), "Number of neurons in calcium dataset does not match number of recorded neurons."

        # Create new maps of neurons to indices
        named_neuron_to_idx = dict()
        unknown_neuron_to_idx = dict()

        # Create masks of which neurons have data
        named_neurons_mask = torch.zeros(302, dtype=torch.bool)
        unknown_neurons_mask = torch.zeros(302, dtype=torch.bool)

        # Create the new calcium data structure
        # len(residual) = len(data) - 1
        standard_calcium_data = torch.zeros(
            max_timesteps, 302, dtype=origin_calcium_data.dtype
        )
        standard_residual_calcium = torch.zeros(
            max_timesteps, 302, dtype=residual_calcium.dtype
        )
        standard_smooth_calcium_data = torch.zeros(
            max_timesteps, 302, dtype=smooth_calcium_data.dtype
        )
        standard_residual_smooth_calcium = torch.zeros(
            max_timesteps, 302, dtype=smooth_residual_calcium.dtype
        )

        # Fill the new calcium data structure with data from named neurons
        slot_to_named_neuron = dict((k, v) for k, v in enumerate(neurons_302))

        for slot, neuron in slot_to_named_neuron.items():
            if neuron in neuron_to_idx:
                # If named neuron is in the dataset
                idx = neuron_to_idx[neuron] # Extract its position in the dataset
                named_neuron_to_idx[neuron] = idx # Include it in the named neuron map

                # Add its data in the neuron standard position
                standard_calcium_data[:, slot] = origin_calcium_data[:, idx]
                standard_residual_calcium[:, slot] = residual_calcium[:, idx] 
                standard_smooth_calcium_data[:, slot] = smooth_calcium_data[:, idx]
                standard_residual_smooth_calcium[:, slot] = smooth_residual_calcium[:, idx]
                named_neurons_mask[slot] = True # Mask the named neuron as having data

        # Randomly distribute the remaining data from unknown neurons
        for neuron in set(neuron_to_idx) - set(named_neuron_to_idx):
            unknown_neuron_to_idx[neuron] = neuron_to_idx[neuron]
        free_slots = list(np.where(~named_neurons_mask)[0])

        slot_to_unknown_neuron = dict(
            zip(
                np.random.choice(free_slots, num_unknown_neurons, replace=False).tolist(),
                list(unknown_neuron_to_idx.keys()),
            )
        )

        for slot, neuron in slot_to_unknown_neuron.items():
            idx = unknown_neuron_to_idx[neuron]
            standard_calcium_data[:, slot] = origin_calcium_data[:, idx]
            standard_residual_calcium[:, slot] = residual_calcium[:, idx]
            standard_smooth_calcium_data[:, slot] = smooth_calcium_data[:, idx]
            standard_residual_smooth_calcium[:, slot] = smooth_residual_calcium[:, idx]
            unknown_neurons_mask[slot] = True

        # Combined slot to neuron mapping
        slot_to_neuron = dict()
        slot_to_neuron.update(slot_to_named_neuron)
        slot_to_neuron.update(slot_to_unknown_neuron)

        # Modify the worm dataset to with new attributes
        single_worm_dataset.update(
            {
                "calcium_data": standard_calcium_data,
                "smooth_calcium_data": standard_smooth_calcium_data,
                "residual_calcium": standard_residual_calcium,
                "smooth_residual_calcium": standard_residual_smooth_calcium,
                "named_neurons_mask": named_neurons_mask,
                "unknown_neurons_mask": unknown_neurons_mask,
                "neurons_mask": named_neurons_mask | unknown_neurons_mask,
                "named_neuron_to_idx": named_neuron_to_idx,
                "idx_to_named_neuron": dict((v, k) for k, v in named_neuron_to_idx.items()),
                "unknown_neuron_to_idx": unknown_neuron_to_idx,
                "idx_to_unknown_neuron": dict(
                    (v, k) for k, v in unknown_neuron_to_idx.items()
                ),
                "slot_to_named_neuron": slot_to_named_neuron,
                "named_neuron_to_slot": dict(
                    (v, k) for k, v in slot_to_named_neuron.items()
                ),
                "slot_to_unknown_neuron": slot_to_unknown_neuron,
                "unknown_neuron_to_slot": dict(
                    (v, k) for k, v in slot_to_unknown_neuron.items()
                ),
                "slot_to_neuron": slot_to_neuron,
                "neuron_to_slot": dict((v, k) for k, v in slot_to_neuron.items()),
            }
        )

        # Delete all original index mappings
        keys_to_delete = [key for key in single_worm_dataset if "idx" in key]
        for key in keys_to_delete:
            single_worm_dataset.pop(key, None)

        # Return the dataset for this worm
        return single_worm_dataset

    def preprocess_pipeline(self):
        """Preprocesses the data for each worm in the all_raw_data.
        """

        # Load the data
        self.load_mat()

        # Auxiliar variables
        worm_counter = -1
        data_dict = dict()

        # Iterate over the .mat files
        for matf, data in self.all_raw_data.items():
            
            # Iterate over worms
            for i, calcium_data in enumerate(data['traces']):

                worm_counter += 1
                worm = "worm" + str(worm_counter) # Worm name

                # Get the IDs of the i-th worm in the file
                ids = [(self._pick_non_none(j) if isinstance(j, list) else j) for j in data['ids'][i]]

                # idx = unique neurons indices, neuron_to_idx = mapping from neuron to index
                ids, calcium_data, neuron_to_idx = self.find_unique(ids, calcium_data)

                # Retrieve the number of neurons that were labeled
                num_named = len(
                    [k for k in neuron_to_idx.keys() if not k.isnumeric()]
                )

                # Reshape time into a column vector and convert to tensor
                time_in_seconds = data['tvs'][i].reshape(data['tvs'][i].shape[0], 1)
                time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32)

                # Apply the transformation to the data
                calcium_data = self.apply_transform(calcium_data)

                # Resample the data to a fixed time step
                time_in_seconds, calcium_data = self.interpolate_data(time_in_seconds, calcium_data)

                # Calculate the time step
                dt = torch.zeros_like(time_in_seconds).to(torch.float32)
                dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]

                # Smooth the data
                smooth_calcium_data, residual, smooth_residual = self.smooth_data_preprocess(
                    calcium_data
                )

                # Update the data
                self.update_non_std_dict(data_dict, worm, calcium_data, smooth_calcium_data,
                                            residual, smooth_residual, neuron_to_idx,
                                            time_in_seconds, num_named)
                
                # Standardize the shape of calcium data to (max_timesteps, num_neurons=302)
                data_dict[worm] = self.reshape_calcium_data(data_dict[worm])

                logging.info('{}/{} - {} done. \tLabeled {},\tUnlabeled {},\tCa shape ({},{})'.format(
                    self.dataset, matf, worm, num_named, len(data_dict[worm]['slot_to_unknown_neuron']),
                    data_dict[worm]['calcium_data'].shape[0], data_dict[worm]['calcium_data'].shape[1]))

        # Pickle the data
        if self.pckl:
            file = os.path.join(processed_path, self.dataset+'.pickle')
            pickle_out = open(file, "wb")
            pickle.dump(data_dict, pickle_out)
            pickle_out.close()
            logging.info('{} saved into /data/processed/neural/{}.pickle'.format(
                        self.dataset, self.dataset))
        
        # Save inside the class
        self.data = data_dict