class Flavell2023Preprocessor(BasePreprocessor):
    def __init__(self, transform, smooth_method, resample_dt):
        super().__init__("Flavell2023", transform, smooth_method, resample_dt)

    def load_data(self, file_name):
        if file_name.endswith(".h5"):
            data = h5py.File(
                os.path.join(self.raw_data_path, self.dataset, file_name), "r"
            )
        elif file_name.endswith(".json"):
            with open(
                os.path.join(self.raw_data_path, self.dataset, file_name), "r"
            ) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_name}")
        return data

    def extract_data(self, file_data):
        if isinstance(file_data, h5py.File):
            time_in_seconds = np.array(file_data["timestamp_confocal"], dtype=float)
            time_in_seconds = (
                time_in_seconds - time_in_seconds[0]
            )  # start time at 0.0 seconds
            time_in_seconds = time_in_seconds.reshape((-1, 1))
            calcium_data = np.array(file_data["trace_array_F20"], dtype=float)
            neurons = np.array(file_data["neuropal_label"], dtype=str)
            neurons_copy = []
            for neuron in neurons:
                if neuron.replace("?", "L") not in set(neurons_copy):
                    neurons_copy.append(neuron.replace("?", "L"))
                else:
                    neurons_copy.append(neuron.replace("?", "R"))
            neurons = np.array(neurons_copy)

        elif isinstance(file_data, dict):  # assuming JSON format

            avg_time = file_data['avg_timestep'] * 60 # average time step in seconds (float)
            raw_traces = file_data['trace_array'] # Raw traces (list)
            max_t = len(raw_traces[0]) # Max time steps (int)
            number_neurons = len(raw_traces) # Number of neurons (int)
            ids = file_data['labeled'] # Labels (list)
        
            time_in_seconds = np.arange(0, max_t*avg_time, avg_time) # Time vector in seconds
            time_in_seconds = time_in_seconds.reshape((-1, 1))

            calcium_data = np.zeros((max_t, number_neurons)) # All traces
            for i, trace in enumerate(raw_traces):
                calcium_data[:, i] = trace

            neurons = [str(i) for i in range(number_neurons)]

            for i in ids.keys():
                label = ids[str(i)]['label']
                neurons[int(i)-1] = label

            # Treat the '?' labels
            for i in range(number_neurons):

                label = neurons[i]

                if not label.isnumeric():
                    if '?' in label:
                        # Find the group which the neuron belongs to
                        label_split = label.split('?')[0]
                        # Verify possible labels
                        possible_labels = [neuron_name for neuron_name in NEURONS_302 if label_split in neuron_name]
                        # Exclude possibilities that we already have
                        possible_labels = [neuron_name for neuron_name in possible_labels if neuron_name not in neurons]
                        # Random pick one of the possibilities
                        neurons[i] = random.choice(possible_labels)
                
            neurons = np.array(neurons)

            neurons, unique_indices = np.unique(neurons, return_index=True, return_counts=False)
            calcium_data = calcium_data[:, unique_indices]  # only get data for unique neurons

        else:
            raise ValueError(f"Unsupported data type: {type(file_data)}")
        
        return time_in_seconds, calcium_data, neurons

    def preprocess(self):
        # load and preprocess data
        preprocessed_data = {}
        for i, file in enumerate(
            os.listdir(os.path.join(self.raw_data_path, self.dataset))
        ):
            if not (file.endswith(".h5") or file.endswith(".json")):
                continue
            worm = "worm" + str(i)
            file_data = self.load_data(file)
            time_in_seconds, calcium_data, neurons = self.extract_data(file_data)

            neuron_to_idx, num_named_neurons = self.create_neuron_idx(neurons)
            calcium_data = self.transform.fit_transform(calcium_data)
            dt = np.gradient(time_in_seconds, axis=0)
            dt[dt == 0] = np.finfo(float).eps
            residual_calcium = np.gradient(calcium_data, axis=0) / dt
            original_time_in_seconds = time_in_seconds.copy()
            time_in_seconds, calcium_data = self.resample_data(
                original_time_in_seconds, calcium_data
            )
            time_in_seconds, residual_calcium = self.resample_data(
                original_time_in_seconds, residual_calcium
            )
            max_timesteps, num_neurons = calcium_data.shape
            smooth_calcium_data = self.smooth_data(calcium_data, time_in_seconds)
            smooth_residual_calcium = self.smooth_data(
                residual_calcium, time_in_seconds
            )
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
