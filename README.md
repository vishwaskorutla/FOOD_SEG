# Food Classification and Segmentation

This repository contains two related projects:

- `food_classification/` — Flutter frontend, Python prediction services, and Firebase Functions for Indian food classification.
- `image_segmentation/` — Flutter frontend, Python segmentation services, and Firebase Functions for food image segmentation.

The root `flutter/` directory is a local Flutter SDK checkout and is not part of the application source. Virtual environments, generated build output, training datasets, and trained model files are also excluded.

## Model and dataset assets

Trained models and datasets are not stored in this source-only repository. To run or deploy model-backed services, provide the corresponding model files at the paths expected by the service code. Training workflows also require their datasets.
