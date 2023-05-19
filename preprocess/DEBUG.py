def pickle_Uzel2022(transform, smooth_method="fft", resample_dt=1.0):
    """
    Pickles the worm neural activity data from Uzel et al 2022., Cell CurrBio 2022,
    A set of hub neurons and non-local connectivity features support global brain dynamics in C. elegans.
    """
    data_dict = dict()
    # load .mat file
    arr = mat73.loadmat(os.path.join(source_path, "Uzel2022", "Uzel_WT.mat"))[
        "Uzel_WT"
    ]  # load .mat file
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["IDs"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr["traces"]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["tv"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for i, real_data in enumerate(all_traces):
        worm = "worm" + str(i)
        i_IDs = [np.array(j).item() for j in all_IDs[i]]
        _, inds = np.unique(
            i_IDs, return_index=True
        )  # only keep indices of unique neuron IDs
        i_IDs = [i_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {
            nid: (str(int(j)) if type(j) != str else j) for nid, j in enumerate(i_IDs)
        }
        neuron_to_idx = {
            nid: (
                name.replace("0", "")
                if not name.endswith("0") and not name.isnumeric()
                else name
            )
            for nid, name in neuron_to_idx.items()
        }
        neuron_to_idx = dict((v, k) for k, v in neuron_to_idx.items())
        time_in_seconds = timeVectorSeconds[i].reshape(timeVectorSeconds[i].shape[0], 1)
        time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32)
        num_named = len(
            [k for k in neuron_to_idx.keys() if not k.isnumeric()]
        )  # number of neurons that were ID'd
        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data)
        real_data = torch.tensor(
            real_data, dtype=torch.float32
        )  # add a feature dimension and convert to tensor
        # resample the data to a fixed time step
        time_in_seconds, real_data = interpolate_data(
            time_in_seconds, real_data, target_dt=resample_dt
        )
        # calulate the time step
        dt = torch.zeros_like(time_in_seconds)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = real_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )
        data_dict.update(
            {
                worm: {
                    "dataset": "Uzel2022",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": int(num_neurons) - num_named,
                },
            }
        )
        # standardize the shape of calcium data to time x 302
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    # pickle the data
    file = os.path.join(processed_path, "Uzel2022.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Uzel2022 = pickle.load(pickle_in)
    print(Uzel2022.keys(), end="\n\n")


def pickle_Flavell2023(transform, smooth_method="fft", resample_dt=1.0):
    """
    Pickles the worm neural activity data from Flavell et al., bioRxiv 2023,
    Brain-wide representations of behavior spanning multiple timescales and states in C. elegans.
    """
    # imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    data_dict = dict()
    data_dir = os.path.join(source_path, "Flavell2023")
    # process all .h5 files in the data directory
    for i, h5_file in enumerate(os.listdir(data_dir)):
        if not h5_file.endswith(".h5"):
            continue
        # each h5 has the data for one (1) worm
        h5_file = os.path.join(data_dir, h5_file)
        worm = "worm" + str(i)
        h5 = h5py.File(h5_file, "r")
        time_in_seconds = torch.tensor(h5["timestamp_confocal"]).to(torch.float32)
        time_in_seconds = time_in_seconds - time_in_seconds[0]  # start at 0
        time_in_seconds = time_in_seconds.reshape((-1, 1))
        if i == 0:
            print(list(h5.keys()), end="\n\n")
        print("num. worms:", 1, end="\n\n")
        # get calcium data for this worm
        calcium_data = np.array(
            h5["trace_array_F20"], dtype=float
        )  # GCaMP neural activity traced normalized by 20th percentile
        # get neuron labels
        neurons = np.array(
            h5["neuropal_label"], dtype=str
        )  # list of full labels (if neuron wasn't labeled the entry is "missing")
        # flip a coin to chose L/R for unsure bilaterally symmetric neurons
        neurons_copy = []
        for neuron in neurons:
            if neuron.replace("?", "L") not in set(neurons_copy):
                neurons_copy.append(neuron.replace("?", "L"))
            else:
                neurons_copy.append(neuron.replace("?", "R"))
        neurons = np.array(neurons_copy)
        # extract neurons with labels
        named_inds = np.where(neurons != "missing")[0]
        num_named = len(named_inds)
        neuron_to_idx = {
            (neuron if idx in named_inds else str(idx)): idx
            for idx, neuron in enumerate(neurons)
        }
        # normalize the data
        sc = transform
        calcium_data = sc.fit_transform(calcium_data)
        calcium_data = torch.tensor(calcium_data, dtype=torch.float32)
        # resample the data to a fixed time step
        time_in_seconds, calcium_data = interpolate_data(
            time_in_seconds, calcium_data, target_dt=resample_dt
        )
        # calculate the time step
        dt = torch.zeros_like(time_in_seconds)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = calcium_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            calcium_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )
        # add worm to data dictionary
        data_dict.update(
            {
                worm: {
                    "dataset": "Flavell2023",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": calcium_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": max_timesteps,
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": num_neurons,
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_neurons - num_named,
                }
            }
        )
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    # pickle the data
    file = os.path.join(processed_path, "Flavell2023.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Flavell2023 = pickle.load(pickle_in)
    print(Flavell2023.keys(), end="\n\n")
    return data_dict


def pickle_Leifer2023(transform, smooth_method="fft", resample_dt=1.0):
    """
    Pickles the worm neural activity data from Randi, ..., Leifer et al.,
    bioRxiv 2023, Neural signal propagation atlas of C. elegans.
    """
    data_dict = dict()
    data_dir = os.path.join(source_path, "Leifer2023")
    files = os.listdir(data_dir)
    num_worms = int(len(files) / 6)  # every worm has 6 txt files

    for i in range(0, num_worms):
        # worm27 doesn't have neuron labels
        if i == 27:
            continue

        if i < 27:
            worm = "worm" + str(i)
        else:
            worm = "worm" + str(i - 1)

        real_data = []
        with open(os.path.join(data_dir, str(i) + "_gcamp.txt"), "r") as f:
            for line in f.readlines():
                cal = list(map(float, line.split(" ")))
                real_data.append(cal)
        real_data = np.array(real_data)  # format: (time, neuron)
        # skip worms with very short recordings
        if real_data.shape[0] < 1000:
            continue

        label_list = []
        with open(os.path.join(data_dir, str(i) + "_labels.txt"), "r") as f:
            for line in f.readlines():
                l = line.strip("\n")
                label_list.append(l)

        # get numbers of neurons and initialize mapping
        num_unnamed = 0
        num_named = real_data.shape[1] - num_unnamed
        label_list = label_list[: real_data.shape[1]]
        neuron_to_idx = dict()

        # compute the time vectoy
        timeVectorSeconds = []
        with open(os.path.join(data_dir, str(i) + "_t.txt"), "r") as f:
            for line in f.readlines():
                l = line.strip("\n")
                timeVectorSeconds.append(str_to_float(l))
        time_in_seconds = np.array(timeVectorSeconds)
        time_in_seconds = torch.tensor(time_in_seconds).to(torch.float32).unsqueeze(1)

        # iterat through labelled neurons
        for j, item in enumerate(label_list):
            previous_list = label_list[:j]
            # if the neuron is unnamed, give it a number larger than 302
            if item == "" or item == "smthng else":
                label_list[j] = str(j + 302)
                num_unnamed += 1
                neuron_to_idx[str(j + 302)] = j
            else:
                # if the neuron is named, and the name is unique, add it to the dictionary
                if item in NEURONS_302 and item not in previous_list:
                    neuron_to_idx[item] = j
                # if the neuron is named, but the name is not unique, give it a number larger than 302
                elif item in NEURONS_302 and item in previous_list:
                    label_list[j] = str(j + 302)
                    num_unnamed += 1
                    neuron_to_idx[str(j + 302)] = j
                else:
                    # if the neuron is recorded without L or R, choose one valid name for it
                    if (
                        str(item + "L") in NEURONS_302
                        and str(item + "L") not in previous_list
                    ):
                        label_list[j] = str(item + "L")
                        neuron_to_idx[str(item + "L")] = j
                    elif (
                        str(item + "R") in NEURONS_302
                        and str(item + "R") not in previous_list
                    ):
                        label_list[j] = str(item + "R")
                        neuron_to_idx[str(item + "R")] = j
                    else:
                        label_list[j] = str(j + 302)
                        num_unnamed += 1
                        neuron_to_idx[str(j + 302)] = j

        sc = transform  # normalize data
        real_data = sc.fit_transform(real_data)
        real_data = torch.tensor(
            real_data, dtype=torch.float32
        )  # add a feature dimension and convert to tensor
        # replace nan and inf with 0
        real_data = torch.nan_to_num(real_data, nan=0.0, posinf=0.0, neginf=0.0)

        # resample the data to a fixed time step
        time_in_seconds, real_data = interpolate_data(
            time_in_seconds, real_data, target_dt=resample_dt
        )
        # calculate the time step
        dt = torch.zeros_like(time_in_seconds)
        dt[1:] = time_in_seconds[1:] - time_in_seconds[:-1]
        # recalculate max_timesteps and num_neurons
        max_timesteps, num_neurons = real_data.shape
        print(
            "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
            % (max_timesteps, num_neurons, num_named),
            end="\n\n",
        )
        # smooth the data
        smooth_real_data, residual, smooth_residual = smooth_data_preprocess(
            real_data,
            time_in_seconds,
            smooth_method,
            dt=np.median(dt),
        )

        data_dict.update(
            {
                worm: {
                    "dataset": "Leifer2023",
                    "smooth_method": smooth_method.upper(),
                    "worm": worm,
                    "calcium_data": real_data,
                    "smooth_calcium_data": smooth_real_data,
                    "residual_calcium": residual,
                    "smooth_residual_calcium": smooth_residual,
                    "neuron_to_idx": neuron_to_idx,
                    "idx_to_neuron": dict((v, k) for k, v in neuron_to_idx.items()),
                    "max_timesteps": int(max_timesteps),
                    "time_in_seconds": time_in_seconds,
                    "dt": dt,
                    "num_neurons": int(num_neurons),
                    "num_named_neurons": num_named,
                    "num_unknown_neurons": num_unnamed,
                },
            }
        )

        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
        data_dict[worm]["num_named_neurons"] = (
            data_dict[worm]["named_neurons_mask"].sum().item()
        )
        data_dict[worm]["num_unknown_neurons"] = (
            data_dict[worm]["num_neurons"] - data_dict[worm]["num_named_neurons"]
        )

    # pickle the data
    file = os.path.join(processed_path, "Leifer2023.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Leifer2023 = pickle.load(pickle_in)
    print(Leifer2023.keys(), end="\n\n")
    return data_dict
