from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os


NEURONS_302 = [ # TODO: Cite source of this list.
            "ADAL", "ADAR", "ADEL", "ADER", "ADFL", "ADFR", "ADLL", "ADLR", "AFDL", "AFDR",
            "AIAL", "AIAR", "AIBL", "AIBR", "AIML", "AIMR", "AINL", "AINR", "AIYL", "AIYR",
            "AIZL", "AIZR", "ALA", "ALML", "ALMR", "ALNL", "ALNR", "AQR", "AS1", "AS10",
            "AS11", "AS2", "AS3", "AS4", "AS5", "AS6", "AS7", "AS8", "AS9", "ASEL", "ASER",
            "ASGL", "ASGR", "ASHL", "ASHR", "ASIL", "ASIR", "ASJL", "ASJR", "ASKL", "ASKR",
            "AUAL", "AUAR", "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER",
            "AVFL", "AVFR", "AVG", "AVHL", "AVHR", "AVJL", "AVJR", "AVKL", "AVKR", "AVL",
            "AVM", "AWAL", "AWAR", "AWBL", "AWBR", "AWCL", "AWCR", "BAGL", "BAGR", "BDUL",
            "BDUR", "CANL", "CANR", "CEPDL", "CEPDR", "CEPVL", "CEPVR", "DA1", "DA2", "DA3",
            "DA4", "DA5", "DA6", "DA7", "DA8", "DA9", "DB1", "DB2", "DB3", "DB4", "DB5",
            "DB6", "DB7", "DD1", "DD2", "DD3", "DD4", "DD5", "DD6", "DVA", "DVB", "DVC",
            "FLPL", "FLPR", "HSNL", "HSNR", "I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5",
            "I6", "IL1DL", "IL1DR", "IL1L", "IL1R", "IL1VL", "IL1VR", "IL2DL", "IL2DR", "IL2L",
            "IL2R", "IL2VL", "IL2VR", "LUAL", "LUAR", "M1", "M2L", "M2R", "M3L", "M3R", "M4",
            "M5", "MCL", "MCR", "MI", "NSML", "NSMR", "OLLL", "OLLR", "OLQDL", "OLQDR",
            "OLQVL", "OLQVR", "PDA", "PDB", "PDEL", "PDER", "PHAL", "PHAR", "PHBL", "PHBR",
            "PHCL", "PHCR", "PLML", "PLMR", "PLNL", "PLNR", "PQR", "PVCL", "PVCR", "PVDL",
            "PVDR", "PVM", "PVNL", "PVNR", "PVPL", "PVPR", "PVQL", "PVQR", "PVR", "PVT",
            "PVWL", "PVWR", "RIAL", "RIAR", "RIBL", "RIBR", "RICL", "RICR", "RID", "RIFL",
            "RIFR", "RIGL", "RIGR", "RIH", "RIML", "RIMR", "RIPL", "RIPR", "RIR", "RIS",
            "RIVL", "RIVR", "RMDDL", "RMDDR", "RMDL", "RMDR", "RMDVL", "RMDVR", "RMED",
            "RMEL", "RMER", "RMEV", "RMFL", "RMFR", "RMGL", "RMGR", "RMHL", "RMHR", "SAADL",
            "SAADR", "SAAVL", "SAAVR", "SABD", "SABVL", "SABVR", "SDQL", "SDQR", "SIADL",
            "SIADR", "SIAVL", "SIAVR", "SIBDL", "SIBDR", "SIBVL", "SIBVR", "SMBDL", "SMBDR",
            "SMBVL", "SMBVR", "SMDDL", "SMDDR", "SMDVL", "SMDVR", "URADL", "URADR", "URAVL",
            "URAVR", "URBL", "URBR", "URXL", "URXR", "URYDL", "URYDR", "URYVL", "URYVR",
            "VA1", "VA10", "VA11", "VA12", "VA2", "VA3", "VA4", "VA5", "VA6", "VA7", "VA8",
            "VA9", "VB1", "VB10", "VB11", "VB2", "VB3", "VB4", "VB5", "VB6", "VB7", "VB8",
            "VB9", "VC1", "VC2", "VC3", "VC4", "VC5", "VC6", "VD1", "VD10", "VD11", "VD12",
            "VD13", "VD2", "VD3", "VD4", "VD5", "VD6", "VD7", "VD8", "VD9"
        ]

# collect all data
print("Collecting data...")

worm_files = sorted(os.listdir("../corr_data"))
print(worm_files)
seperated_files = [[]]

curr_set = 0

for file in worm_files:
    file_set = int(file.split("_")[1])
    if file_set != curr_set:
        curr_set = file_set
        seperated_files += [[file]]
    else:
        seperated_files[-1] += [file]

for set_idx, files in enumerate(seperated_files):
    if set_idx != 6:
        all_data = np.empty((len(files), 302, 302, 100))
        for i, file in tqdm(enumerate(files)):
            all_data[i] = np.load(f'../corr_data/{file}')

        # filter data
        # length of filtered data list is 302 neurons*302 neurons
        print("Filtering data...")

        filtered_data = [None for i in range(302**2)]

        for i in tqdm(range(all_data.shape[1])):
            for j in range(all_data.shape[2]):
                for worm_idx in range(all_data.shape[0]):
                    if filtered_data[i*all_data.shape[1]+j] is None and not np.any(np.isnan(all_data[worm_idx, i, j])):
                        filtered_data[i*all_data.shape[1]+j] = [all_data[worm_idx, i, j]]
                    elif not np.any(np.isnan(all_data[worm_idx, i, j])):
                        filtered_data[i*all_data.shape[1]+j] += [all_data[worm_idx, i, j]]

        for i in range(len(filtered_data)):
            if filtered_data[i] is not None:
                filtered_data[i] = np.array(filtered_data[i])

        # generate plots 
        print("Generating plots...")
            
        for i, neuron1 in enumerate(NEURONS_302):
            if i > -1:
                for j, neuron2 in tqdm(enumerate(NEURONS_302)):
                    if filtered_data[i*len(NEURONS_302)+j] is not None:
                        plt.figure()
                        plt.ylim(-1, 1)
                        
                        plt.title(f"Dataset {set_idx}: {neuron1}_{neuron2}")
                        plt.xlabel("Lag")
                        plt.ylabel("Correlation Score")
                        
                        mean = np.mean(filtered_data[i*len(NEURONS_302)+j], axis=0)
                        std = np.std(filtered_data[i*len(NEURONS_302)+j], axis=0)

                        plt.plot(np.arange(0, 100, 1), mean, linestyle='dashed', marker='v', alpha=1.0)

                        for k in filtered_data[i*len(NEURONS_302)+j]:
                            plt.plot(np.arange(0, 100, 1), k, alpha=0.3)

                        z=1
                        plt.fill_between(np.arange(0, 100, 1), mean-(std*z), mean+(std*z), alpha=0.4)
                        
                        if not os.path.exists(f"../corr_figs/{set_idx}"):
                            os.makedirs(f"../corr_figs/{set_idx}")
                            
                        if not os.path.exists(f"../corr_figs/{set_idx}/{neuron1}"):
                            os.makedirs(f"../corr_figs/{set_idx}/{neuron1}")
                            
                        plt.savefig(f"../corr_figs/{set_idx}/{neuron1}/{neuron1}_{neuron2}.png")
                        plt.close()
                        