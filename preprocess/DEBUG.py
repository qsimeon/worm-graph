
def pickle_Nichols2017(transform, smooth_method="fft", resample_dt=1.0):
    """
    Pickles the worm neural activity data from Nichols et al., Science 2017,
    A global brain state underlies C. elegans sleep behavior.
    """
    data_dict = dict()

    # 'n2_let'
    # load the first .mat file
    arr = mat73.loadmat(os.path.join(source_path, "Nichols2017", "n2_let.mat"))[
        "n2_let"
    ]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["IDs"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr["traces"]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["timeVectorSeconds"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for i, real_data in enumerate(all_traces):
        worm = "worm" + str(i)
        i_IDs = [(j[0] if isinstance(j, list) else j) for j in all_IDs[i]]
        i_IDs = [
            (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
            for _, j in enumerate(i_IDs)
        ]
        _, inds = np.unique(
            i_IDs, return_index=True
        )  # only keep indices of unique neuron IDs
        i_IDs = [i_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(i_IDs)
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
                    "dataset": "Nichols2017",
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
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])

    # 'n2_prelet'
    # load the second .mat file
    arr = mat73.loadmat(os.path.join(source_path, "Nichols2017", "n2_prelet.mat"))[
        "n2_prelet"
    ]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["IDs"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr["traces"]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["timeVectorSeconds"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for ii, real_data in enumerate(all_traces):
        worm = "worm" + str(ii + i + 1)
        ii_IDs = [(j[0] if isinstance(j, list) else j) for j in all_IDs[ii]]
        ii_IDs = [
            (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
            for _, j in enumerate(ii_IDs)
        ]
        _, inds = np.unique(
            ii_IDs, return_index=True
        )  # only keep indices of unique neuron IDs
        ii_IDs = [ii_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(ii_IDs)
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
        time_in_seconds = timeVectorSeconds[ii].reshape(
            timeVectorSeconds[ii].shape[0], 1
        )
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
                    "dataset": "Nichols2017",
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
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])

    # 'npr1_let'
    # load the third .mat file
    arr = mat73.loadmat(os.path.join(source_path, "Nichols2017", "npr1_let.mat"))[
        "npr1_let"
    ]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["IDs"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr["traces"]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["timeVectorSeconds"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for iii, real_data in enumerate(all_traces):
        worm = "worm" + str(iii + ii + 1 + i + 1)
        iii_IDs = [(j[0] if isinstance(j, list) else j) for j in all_IDs[iii]]
        iii_IDs = [
            (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
            for _, j in enumerate(iii_IDs)
        ]
        _, inds = np.unique(
            iii_IDs, return_index=True
        )  # only keep indices of unique neuron IDs
        iii_IDs = [iii_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(iii_IDs)
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
        time_in_seconds = timeVectorSeconds[iii].reshape(
            timeVectorSeconds[iii].shape[0], 1
        )
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
                    "dataset": "Nichols2017",
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
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])

    # 'npr1_prelet'
    # load the fourth .mat file
    arr = mat73.loadmat(os.path.join(source_path, "Nichols2017", "npr1_prelet.mat"))[
        "npr1_prelet"
    ]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["IDs"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr["traces"]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["timeVectorSeconds"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for iv, real_data in enumerate(all_traces):
        worm = "worm" + str(iv + iii + 1 + ii + 1 + i + 1)
        iv_IDs = [(j[0] if isinstance(j, list) else j) for j in all_IDs[iv]]
        iv_IDs = [
            (str(_) if j is None or isinstance(j, np.ndarray) else str(j))
            for _, j in enumerate(iv_IDs)
        ]
        _, inds = np.unique(
            iv_IDs, return_index=True
        )  # only keep indices of unique neuron IDs
        iv_IDs = [iv_IDs[_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {
            nid: (str(nid) if (j is None or isinstance(j, np.ndarray)) else str(j))
            for nid, j in enumerate(iv_IDs)
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
        time_in_seconds = timeVectorSeconds[iv].reshape(
            timeVectorSeconds[iv].shape[0], 1
        )
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
                    "dataset": "Nichols2017",
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
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    # pickle the data
    file = os.path.join(processed_path, "Nichols2017.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Nichols2017 = pickle.load(pickle_in)
    print(Nichols2017.keys(), end="\n\n")


def pickle_Nguyen2017(transform, smooth_method="fft", resample_dt=1.0):
    """
    Pickles the worm neural activity data from Nguyen et al., PLOS CompBio 2017,
    Automatically tracking neurons in a moving and deforming brain.
    """
    imputer = SimpleImputer(missing_values=np.nan, strategy="median")

    # WORM 0
    # load .mat file for  worm 0
    arr0 = loadmat(
        os.path.join(source_path, "Nguyen2017", "heatData_worm0.mat")
    )  # load .mat file
    print(list(arr0.keys()), end="\n\n")
    # get data for worm 0
    G2 = arr0[
        "G2"
    ]  # the ratio signal is defined as gPhotoCorr/rPhotoCorr, the Ratio is then normalized as delta R/ R0. is the same way as R2 and G2.
    cgIdx = arr0[
        "cgIdx"
    ].squeeze()  # ordered indices derived from heirarchically clustering the correlation matrix.
    real_data0 = G2[cgIdx - 1, :].T  # to show organized traces, use Ratio2(cgIdx,:)
    real_data0 = imputer.fit_transform(real_data0)  # impute missing values (i.e. NaNs)
    # time vector
    time_in_seconds0 = arr0.get("hasPointsTime", np.arange(real_data0.shape[0]))
    time_in_seconds0 = time_in_seconds0.reshape(-1, 1)
    time_in_seconds0 = torch.tensor(time_in_seconds0).to(torch.float32)
    num_named0 = 0
    worm0_ID = {i: str(i) for i in range(real_data0.shape[1])}
    worm0_ID = dict((v, k) for k, v in worm0_ID.items())
    # normalize the data
    sc = transform
    real_data0 = sc.fit_transform(real_data0)
    real_data0 = torch.tensor(real_data0, dtype=torch.float32)
    # resample the data to a fixed time step
    time_in_seconds0, real_data0 = interpolate_data(
        time_in_seconds0, real_data0, target_dt=resample_dt
    )
    # calculate the time step
    dt0 = torch.zeros_like(time_in_seconds0)
    dt0[1:] = time_in_seconds0[1:] - time_in_seconds0[:-1]
    # recalculate max_timesteps and num_neurons
    max_time0, num_neurons0 = real_data0.shape
    print(
        "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
        % (max_time0, num_neurons0, num_named0),
        end="\n\n",
    )

    # WORM 1
    # load .mat file for  worm 1
    arr1 = loadmat(
        os.path.join(source_path, "Nguyen2017", "heatData_worm1.mat")
    )  # load .mat file
    print(list(arr1.keys()), end="\n\n")
    # get data for worm 1
    G2 = arr1[
        "G2"
    ]  # the ratio signal is defined as gPhotoCorr/rPhotoCorr, the Ratio is then normalized as delta R/ R0. is the same way as R2 and G2.
    cgIdx = arr1[
        "cgIdx"
    ].squeeze()  # ordered indices derived from heirarchically clustering the correlation matrix.
    real_data1 = G2[cgIdx - 1, :].T  # to show organized traces, use Ratio2(cgIdx,:)
    real_data1 = imputer.fit_transform(real_data1)  # replace NaNs
    # time vector
    time_in_seconds1 = arr1.get("hasPointsTime", np.arange(real_data1.shape[0]))
    time_in_seconds1 = time_in_seconds1.reshape(-1, 1)
    time_in_seconds1 = torch.tensor(time_in_seconds1).to(torch.float32)
    num_named1 = 0
    worm1_ID = {i: str(i) for i in range(real_data1.shape[1])}
    worm1_ID = dict((v, k) for k, v in worm1_ID.items())
    # normalize the data
    sc = transform
    real_data1 = sc.fit_transform(real_data1)
    real_data1 = torch.tensor(real_data1, dtype=torch.float32)
    # resample the data to a fixed time step
    time_in_seconds1, real_data1 = interpolate_data(
        time_in_seconds1, real_data1, target_dt=resample_dt
    )
    # calculate the time step
    dt1 = torch.zeros_like(time_in_seconds1)
    dt1[1:] = time_in_seconds1[1:] - time_in_seconds1[:-1]
    # recalculate max_timesteps and num_neurons
    max_time1, num_neurons1 = real_data1.shape
    print(
        "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
        % (max_time1, num_neurons1, num_named1),
        end="\n\n",
    )

    # WORM 2
    # load .mat file for  worm 1
    arr2 = loadmat(
        os.path.join(source_path, "Nguyen2017", "heatData_worm2.mat")
    )  # load .mat file
    print(list(arr2.keys()), end="\n\n")
    # get data for worm 2
    G2 = arr2[
        "G2"
    ]  # the ratio signal is defined as gPhotoCorr/rPhotoCorr, the Ratio is then normalized as delta R/ R0. is the same way as R2 and G2.
    cgIdx = arr2[
        "cgIdx"
    ].squeeze()  # ordered indices derived from heirarchically clustering the correlation matrix.
    real_data2 = G2[cgIdx - 1, :].T  # to show organized traces, use Ratio2(cgIdx,:)
    real_data2 = imputer.fit_transform(real_data2)  # replace NaNs
    # time vector
    time_in_seconds2 = arr2.get("hasPointsTime", np.arange(real_data2.shape[0]))
    time_in_seconds2 = time_in_seconds2.reshape(-1, 1)
    time_in_seconds2 = torch.tensor(time_in_seconds2).to(torch.float32)
    num_named2 = 0
    worm2_ID = {i: str(i) for i in range(real_data2.shape[1])}
    worm2_ID = dict((v, k) for k, v in worm2_ID.items())
    # normalize the data
    sc = transform
    real_data2 = sc.fit_transform(real_data2)
    real_data2 = torch.tensor(real_data2, dtype=torch.float32)
    # resample the data to a fixed time step
    time_in_seconds2, real_data2 = interpolate_data(
        time_in_seconds2, real_data2, target_dt=resample_dt
    )
    # calculate the time step
    dt2 = torch.zeros_like(time_in_seconds2)
    dt2[1:] = time_in_seconds2[1:] - time_in_seconds2[:-1]
    # recalculate max_timesteps and num_neurons
    max_time2, num_neurons2 = real_data2.shape
    print(
        "len. Ca recording %s, total num. neurons %s, num. ID'd neurons %s"
        % (max_time2, num_neurons2, num_named2),
        end="\n\n",
    )
    # smooth the data
    smooth_real_data0, residual0, smooth_residual0 = smooth_data_preprocess(
        real_data0,
        time_in_seconds0,
        smooth_method,
        dt=np.meadian(dt0),
    )
    smooth_real_data1, residual1, smooth_residual1 = smooth_data_preprocess(
        real_data1,
        time_in_seconds1,
        smooth_method,
        dt=np.median(dt1),
    )
    smooth_real_data2, residual2, smooth_residual2 = smooth_data_preprocess(
        real_data2,
        time_in_seconds2,
        smooth_method,
        dt=np.median(dt2),
    )
    # pickle the data
    data_dict = {
        "worm0": {
            "dataset": "Nguyen2017",
            "smooth_method": smooth_method.upper(),
            "worm": "worm0",
            "calcium_data": real_data0,
            "smooth_calcium_data": smooth_real_data0,
            "residual_calcium": residual0,
            "smooth_residual_calcium": smooth_residual0,
            "neuron_to_idx": worm0_ID,
            "idx_to_neuron": dict((v, k) for k, v in worm0_ID.items()),
            "max_timesteps": max_time0,
            "time_in_seconds": time_in_seconds0,
            "dt": dt0,
            "num_neurons": num_neurons0,
            "num_named_neurons": num_named0,
            "num_unknown_neurons": num_neurons0 - num_named0,
        },
        "worm1": {
            "dataset": "Nguyen2017",
            "smooth_method": smooth_method.upper(),
            "worm": "worm1",
            "calcium_data": real_data1,
            "smooth_calcium_data": smooth_real_data1,
            "residual_calcium": residual1,
            "smooth_residual_calcium": smooth_residual1,
            "neuron_to_idx": worm1_ID,
            "idx_to_neuron": dict((v, k) for k, v in worm1_ID.items()),
            "max_timesteps": max_time1,
            "time_in_seconds": time_in_seconds1,
            "dt": dt1,
            "num_neurons": num_neurons1,
            "num_named_neurons": num_named1,
            "num_unknown_neurons": num_neurons1 - num_named1,
        },
        "worm2": {
            "dataset": "Nguyen2017",
            "smooth_method": smooth_method.upper(),
            "worm": "worm2",
            "calcium_data": real_data2,
            "smooth_calcium_data": smooth_real_data2,
            "residual_calcium": residual2,
            "smooth_residual_calcium": smooth_residual2,
            "neuron_to_idx": worm2_ID,
            "idx_to_neuron": dict((v, k) for k, v in worm2_ID.items()),
            "max_timesteps": max_time2,
            "time_in_seconds": time_in_seconds2,
            "dt": dt2,
            "num_neurons": num_neurons2,
            "num_named_neurons": num_named2,
            "num_unknown_neurons": num_neurons2 - num_named2,
        },
    }
    for worm in data_dict.keys():
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    file = os.path.join(processed_path, "Nguyen2017.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Nguyen2017 = pickle.load(pickle_in)
    print(Nguyen2017.keys(), end="\n\n")


def pickle_Kaplan2020(transform, smooth_method="fft", resample_dt=1.0):
    """
    Pickles the worm neural activity data from Kaplan et al., Neuron 2020,
    Nested Neuronal Dynamics Orchestrate a Behavioral Hierarchy across Timescales.
    """
    data_dict = dict()

    # 'RIShisCl_Neuron2019'
    # load the first .mat file
    arr = mat73.loadmat(
        os.path.join(source_path, "Kaplan2020", "Neuron2019_Data_RIShisCl.mat")
    )["RIShisCl_Neuron2019"]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["neuron_ID"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr[
        "traces_bleach_corrected"
    ]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["time_vector"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for i, real_data in enumerate(all_traces):
        worm = "worm" + str(i)
        _, inds = np.unique(
            all_IDs[i], return_index=True
        )  # only keep indices of unique neuron IDs
        all_IDs[i] = [all_IDs[i][_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {nid: str(j) for nid, j in enumerate(all_IDs[i])}
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
                    "dataset": "Kaplan2020",
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
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])

    # 'MNhisCl_RIShisCl_Neuron2019'
    # load the second .mat file
    arr = mat73.loadmat(
        os.path.join(source_path, "Kaplan2020", "Neuron2019_Data_MNhisCl_RIShisCl.mat")
    )["MNhisCl_RIShisCl_Neuron2019"]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["neuron_ID"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr[
        "traces_bleach_corrected"
    ]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["time_vector"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for ii, real_data in enumerate(all_traces):
        worm = "worm" + str(ii + i + 1)
        _, inds = np.unique(
            all_IDs[ii], return_index=True
        )  # only keep indices of unique neuron IDs
        all_IDs[ii] = [all_IDs[ii][_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {nid: str(j) for nid, j in enumerate(all_IDs[ii])}
        neuron_to_idx = {
            nid: (
                name.replace("0", "")
                if not name.endswith("0") and not name.isnumeric()
                else name
            )
            for nid, name in neuron_to_idx.items()
        }
        neuron_to_idx = dict((v, k) for k, v in neuron_to_idx.items())
        time_in_seconds = timeVectorSeconds[ii].reshape(
            timeVectorSeconds[ii].shape[0], 1
        )
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
                    "dataset": "Kaplan2020",
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
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])

    # 'MNhisCl_RIShisCl_Neuron2019'
    # load the third .mat file
    arr = mat73.loadmat(
        os.path.join(source_path, "Kaplan2020", "Neuron2019_Data_SMDhisCl_RIShisCl.mat")
    )["SMDhisCl_RIShisCl_Neuron2019"]
    print(list(arr.keys()), end="\n\n")
    # get data for all worms
    all_IDs = arr["neuron_ID"]  # identified neuron IDs (only subset have neuron names)
    all_traces = arr[
        "traces_bleach_corrected"
    ]  # neural activity traces corrected for bleaching
    timeVectorSeconds = arr["time_vector"]
    print("num. worms:", len(all_IDs), end="\n\n")
    for iii, real_data in enumerate(all_traces):
        worm = "worm" + str(iii + ii + 1 + i + 1)
        _, inds = np.unique(
            all_IDs[iii], return_index=True
        )  # only keep indices of unique neuron IDs
        all_IDs[iii] = [all_IDs[iii][_] for _ in inds]
        real_data = real_data[:, inds.astype(int)]  # only get data for unique neurons
        neuron_to_idx = {nid: str(j) for nid, j in enumerate(all_IDs[iii])}
        neuron_to_idx = {
            nid: (
                name.replace("0", "")
                if not name.endswith("0") and not name.isnumeric()
                else name
            )
            for nid, name in neuron_to_idx.items()
        }
        neuron_to_idx = dict((v, k) for k, v in neuron_to_idx.items())
        time_in_seconds = timeVectorSeconds[iii].reshape(
            timeVectorSeconds[iii].shape[0], 1
        )
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
                    "dataset": "Kaplan2020",
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
        # standardize the shape of calcium data to 302 x time
        data_dict[worm] = reshape_calcium_data(data_dict[worm])
    # pickle the data
    file = os.path.join(processed_path, "Kaplan2020.pickle")
    pickle_out = open(file, "wb")
    pickle.dump(data_dict, pickle_out)
    pickle_out.close()
    pickle_in = open(file, "rb")
    Kaplan2020 = pickle.load(pickle_in)
    print(Kaplan2020.keys(), end="\n\n")


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
