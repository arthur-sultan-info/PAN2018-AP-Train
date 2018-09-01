
###  PAN2018-AP-Train
This repository contains the code to train the gender classifier for the PAN 2018 AP task.

## Requirements

This tool is coded in Python 3 and has been tested for Python 3.5.5

Required additionnal libraries:

| Library      | Version |
|--------------|---------|
| numpy        | 1.11.X  |
| scikit-learn | 0.18.X  |     
| pandas       | 0.19.X  |       
| nltk         | 3.2.X   |
| opencv       | 3.4.X   |
| tensorflow   | 1.5.X   |

You also need to install [darkflow](https://github.com/thtrieu/darkflow).
Please also download [yolo weights](https://pjreddie.com/media/files/yolov2.weights) and put it in feature_extractors/bin.


## Extracting features
In order to train the classifiers, you first need to extract the features. To do so, just run the corresponding python feature extractor file (e.g faceRecognition_extractor.py).

The features should be extracted from the [PAN 2018 training dataset](https://s3.amazonaws.com/autoritas.pan/pan18-author-profiling-training-2018-02-27.zip). Download it, and put it in the folder of your choice. Then, make sure the global variable DEFAULT_DATASET_PATH in feature_extractors/consts.py point to this directory.

To extract all features, run the following command:
```
cd feature_extractors
python faceRecognition_extractor.py && python globalFeatures_extractor.py && python yolo_extractor.py
```
Features are extracted in the directory: feature_extractors/extracted-features.

## Training classifiers
To train a classifier, just run the corresponding python train file (e.g train-face-recognition.py).
To train all classifiers, including the final classifier, run the following command:
```
cd train_classifiers
python train-color-histogram.py && python train-face-recognition.py && python train-lbp.py && python train-object-detection.py && python train-meta-image.py && python train-aggregation-classifier.py && python train-final-clf-text-image.py
```
Classifiers are saved in the directory: train_classifiers/trained-classifiers.
