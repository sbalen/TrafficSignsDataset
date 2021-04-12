"""TrafficSignDataset dataset."""
from glob import glob
import tensorflow_datasets as tfds
import cv2
import os.path

import uuid as uuid

_DESCRIPTION = """
This is a set of clearly cropped images of (Belgian) traffic signs sorted in to one of 62 classes. More info on <https://btsd.ethz.ch/shareddata/index.html>.

NOTE: The images are scaled to 32*32, native resolution mostly higher, aspect ratio does differ some distortion occurs
"""
_CITATION = r"""
@inproceedings{inproceedings,
                author = {Mathias, Markus and Timofte, Radu and Benenson, Rodrigo and Van Gool, Luc},
                year = {2013},
                month = {08},
                ages = {1-8},
                title = {Traffic sign recognition — How far are we from the solution?},
                isbn = {978-1-4673-6128-6},
                doi = {10.1109/IJCNN.2013.6707049}}
"""
_DL_URLS = {
    "train_data_link": "https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip",
    "test_data_link": "https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip"
}

class Trafficsignsdataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for TrafficSignsDataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(32, 32, 3)),
            'label': tfds.features.ClassLabel(num_classes=62)
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://btsd.ethz.ch/shareddata/index.html',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract(_DL_URLS)
    #test_path = dl_manager.download_and_extract(_DL_URLS["test_data_link"])
    examples =  self._generate_examples(path=path)
    return {
        'train': self._generate_examples(path=path["train_data_link"]),
        'test': self._generate_examples(path=path["test_data_link"]),
    }

  def _generate_examples(self, path):
    """Yields examples."""

    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Loop through extracted dirs in the current data_dir.
    for img_path in path.glob("*/*/*.ppm"):
        yield str(uuid.uuid4()), {
            'image': cv2.resize(cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB), (32,32)),
            'label': str(img_path).split('/')[-2]
        }

