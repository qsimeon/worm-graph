# Built-in libraries
import logging
import os
import pickle
import shutil
import unittest
import warnings
from zipfile import ZipFile

# Third-party libraries
import requests
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler

# Local libraries
from preprocess import _utils as utils

# Filter warnings and debug messages
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)


class TestPreprocessors(unittest.TestCase):
    def setUp(self):
        self.processed_data_path = "/om2/user/qsimeon/worm-graph/data/processed/neural"
        # Load config file
        config = OmegaConf.load("conf/preprocess.yaml")
        preprocess_config = config.preprocess

        # Download and extract zipfile
        url = preprocess_config.url
        zip_filename = preprocess_config.zipfile
        dir_name = zip_filename.rsplit(".", 1)[
            0
        ]  # Strip off ".zip" to get directory name
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(zip_filename, "wb") as f:
                f.write(response.content)
            with ZipFile(zip_filename, "r") as zip_ref:
                zip_ref.extractall(dir_name)
        else:
            raise Exception(f"Failed to download zipfile from {url}")

        return preprocess_config, zip_filename, dir_name

    def tearDown(self):
        _, zip_filename, dir_name = self.setUp()
        # Remove directory and zipfile after tests
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        if os.path.exists(zip_filename):
            os.remove(zip_filename)

    def test_preprocessors(self):
        for dataset in utils.VALID_DATASETS:
            # Preprocess dataset
            preprocessor_class = getattr(utils, f"{dataset}Preprocessor")
            preprocessor = preprocessor_class(
                transform=StandardScaler(), smooth_method="FFT", resample_dt=0.5
            )

            # Check preprocess method
            try:
                preprocessor.preprocess()
            except Exception as e:
                self.fail(
                    f"preprocess method raised exception {e} for {dataset}Preprocessor"
                )

            # Check data loading and extraction
            with open(
                os.path.join(self.processed_data_path, f"{dataset}.pickle"), "rb"
            ) as f:
                data = pickle.load(f)

            # Test if keys 'worm0', 'worm1', etc. are in data
            for i in range(1):  # assuming we have at least 1 worms in test data
                self.assertIn(f"worm{i}", data.keys())

            # Test data extraction for each worm
            for worm_key in data.keys():
                worm_data = data[worm_key]
                self.assertIn("neuron_to_slot", worm_data.keys())
                self.assertIn("time_in_seconds", worm_data.keys())
                self.assertIn("smooth_calcium_data", worm_data.keys())
                self.assertIn("slot_to_named_neuron", worm_data.keys())

                calcium_traces = worm_data["smooth_calcium_data"]
                self.assertIsInstance(calcium_traces, torch.Tensor)
                self.assertEqual(len(calcium_traces.shape), 2)  # should be 2D

                time_vector = worm_data["time_in_seconds"]
                self.assertIsInstance(time_vector, torch.Tensor)
                self.assertEqual(
                    time_vector.dtype, torch.float
                )  # should be torch.float32
                self.assertEqual(len(time_vector.shape), 2)  # should be 2D


if __name__ == "__main__":
    unittest.main()
