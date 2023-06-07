class Flavell2023Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, resample_dt):
        super().__init__("Flavell2023", transform, smooth_method, resample_dt)

    def load_data(self, file_name):
        file_name = os.path.join(self.raw_data_path, self.dataset, file_name)
        with open(file_name, 'r') as file:
            return json.load(file)
        
    def extract_data(self, data):

        avg_time = data['avg_timestep'] * 60 # average time step in seconds (float)
        max_t = data['max_t'] # Max time steps (float)
        raw_traces = data['trace_array'] # Raw traces (list)
        ids = data['labeled'] # Labels (list)
        
        timeVectorSeconds = np.arange(0, max_t*avg_time, avg_time) # Time vector in seconds
        
        all_traces = np.zeros((max_t, len(raw_traces))) # All traces
        for i, trace in enumerate(raw_traces):
            all_traces[:,i] = trace
        
        # We need to organize the data such as the traces are in the same order as the IDs

        neuron_IDs = [str(i) for i in range(len(raw_traces))]

        for i in ids.keys():
            label = ids[str(i)]['label']
            neuron_IDs[int(i)-1] = label

        # Treat the '?' labels
        for i in range(len(neuron_IDs)):

            label = neuron_IDs[i]

            if not label.isnumeric():

                if '?' in label:
                    # Find the group which the neuron belongs to
                    label_split = label.split('?')[0]
                    # Verify possible labels
                    possible_labels = [neuron_name for neuron_name in NEURONS_302 if label_split in neuron_name]
                    # Exclude possibilities that we already have
                    possible_labels = [neuron_name for neuron_name in possible_labels if neuron_name not in neuron_IDs]
                    # Random pick one of the possibilities
                    neuron_IDs[i] = random.choice(possible_labels)

        # Returning the neuron IDs in a list => each element corresponds to a column of traces (num_neurons)
        # Returning the traces in a matrix => each row is a time step and each column is a neuron (max_t, num_neurons)
        # Returning the time vector => each element is a time step in seconds (max_t, )
        
        return neuron_IDs, all_traces, timeVectorSeconds
        
    def preprocess(self):
        # load and preprocess data
        preprocessed_data = {}
        worm_idx = 0  # Initialize worm index outside file loop
        data_dir = os.path.join(self.raw_data_path, self.dataset)

        for file_name in os.listdir(data_dir): # Each file is a single worm

            data = self.load_data(file_name)
            neuron_IDs, trace_data, raw_timeVectorSeconds = self.extract_data(data)

            worm = "worm" + str(worm_idx)  # Use global worm index
            worm_idx += 1  # Increment worm index
            unique_IDs, unique_indices = np.unique(neuron_IDs, return_index=True, return_counts=False)
            trace_data = trace_data[:, unique_indices]  # only get data for unique neurons
            neuron_to_idx, num_named_neurons = self.create_neuron_idx(unique_IDs)
            time_in_seconds = raw_timeVectorSeconds.reshape(
                raw_timeVectorSeconds.shape[0], 1
            )
            time_in_seconds = np.array(time_in_seconds, dtype=np.float32)

            if self.transform is not None:
                calcium_data = self.normalize_data(trace_data)
            else:
                calcium_data = trace_data

            dt = np.gradient(time_in_seconds, axis=0)
            dt[dt == 0] = np.finfo(float).eps
            residual_calcium = np.gradient(calcium_data, axis=0) / dt
            original_time_in_seconds = time_in_seconds.copy()

            if dt is not None:
                time_in_seconds, calcium_data = self.resample_data(
                    original_time_in_seconds, calcium_data
                )
                time_in_seconds, residual_calcium = self.resample_data(
                    original_time_in_seconds, residual_calcium
                )

            max_timesteps, num_neurons = calcium_data.shape

            if self.smooth_method is not None:
                smooth_calcium_data = self.smooth_data(
                    calcium_data, original_time_in_seconds
                )
                smooth_residual_calcium = self.smooth_data(
                    residual_calcium, time_in_seconds
                )
            else:
                smooth_calcium_data = calcium_data # TODO: Return None => modify reshape_calcium_data
                smooth_residual_calcium = calcium_data

            num_unknown_neurons = int(num_neurons) - num_named_neurons
            worm_dict = {
                worm: {
                    "dataset": self.dataset,
                    "smooth_method": self.smooth_method,
                    "worm": worm,
                    "calcium_data": calcium_data,
                    "smooth_calcium_data": smooth_calcium_data,
                    "residual_calcium": residual_calcium,
                    "smooth_residual_calcium": smooth_residual_calcium,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named_neurons,
                    "num_unknown_neurons": num_unknown_neurons,
                }
            }
            preprocessed_data.update(worm_dict)
        # reshape calcium data
        for worm in preprocessed_data.keys():
            preprocessed_data[worm] = reshape_calcium_data(preprocessed_data[worm])
        # save data
        self.save_data(preprocessed_data)
        print(f"Finished processing {self.dataset}!", end="\n\n")