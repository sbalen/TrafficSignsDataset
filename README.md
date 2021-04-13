# TrafficSignsDataset
A Tensorflow Dataset containing cropped images of traffic signs via: https://btsd.ethz.ch/shareddata/index.html

## Installation
To install directly from github:

```
  pip install git+https://github.com/sbalen/TrafficSignsDataset.git
```

To install if you want to make changes to the code:

```
  git clone https://github.com/sbalen/TrafficSignsDataset.git
  cd TrafficSignsDataset
  pip install -e .
```

## Requirements
You'll need to rebuild tfds after installation prior to use
```
  tfds build
```

## Example usage
```
  import tensorflow_datasets as tfds
  dataset, info = tfds.load("Trafficsigndataset", split="train", with_info=True)
```
