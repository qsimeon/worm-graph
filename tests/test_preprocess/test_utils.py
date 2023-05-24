import unittest
import torch
import os
import warnings
import pickle
from preprocess import _utils as utils
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestPreprocessors(unittest.TestCase):
    def setUp(self):
        self.processed_data_path = (
            "/Users/quileesimeon/GitHub Repos/worm-graph/data/processed/neural"
        )

    def test_preprocessors(self):
        for dataset in utils.VALID_DATASETS:
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
                self.assertIn("smooth_calcium_data", worm_data.keys())
                self.assertIn("slot_to_named_neuron", worm_data.keys())

                calcium_traces = worm_data["smooth_calcium_data"]
                self.assertIsInstance(calcium_traces, torch.Tensor)
                self.assertEqual(len(calcium_traces.shape), 2)  # should be 2D


if __name__ == "__main__":
    unittest.main()
