"""TrafficSignDataset dataset."""

import tensorflow_datasets as tfds
from . import TrafficSignsDataset

class TrafficsignsdatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for TrafficSignsDataset dataset."""
  DATASET_CLASS = TrafficSignsDataset.Trafficsignsdataset
  SPLITS = {
    'train': 1,
    'test': 1,
  }
  SKIP_CHECKSUMS = True

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  DL_EXTRACT_RESULT = {
    "train_data_link": "train",
    "test_data_link": "test"
}

if __name__ == '__main__':
  tfds.testing.test_main()
