# Capstone Project for Udacity Machine Learning Nano Degree

## Dataset

The dataset to be used contains 25,000 images of dogs and cats and can be obtained in the Kaggle's Competition site: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data.

The folder structure should be modified following the steps in "notebooks/01. Data loading and analysis.ipynb"

Folder structure:

data/input/CLASS
data/valid/CLASS
data/test/CLASS

## Benchmark model

The benchmark model code can be found in "notebook/02 - Simple CNN.ipynb" notebook. The code to load the model, train it and test it can be found there.

## Pre-trained models

### VGG16

To train the model execute:

```bash
$ python3 vgg16_train.py
```

To fine tune the model execute:
```bash
$ python3 vgg16_finetune.py
```

The model can be test it using the section called "Test the model" in the notebook "notebooks/03 - VGG16 pre-trained model.ipynb". The code from vgg16_train.py and vgg16_finetune.py is included in the notebook but it was not used from the notebook.


### ResNet50

To train the model execute:

```bash
$ python3 resnet50_train.py
```

To fine tune the model execute:
```bash
$ python3 resnet50_finetune.py
```

The model can be test it using the section called "Test the model" in the notebook "notebooks/04 - Resnet50 pre-trained model.ipynb". The code from resnet50_train.py and resnet50_finetune.py is included in the notebook but it was not used from the notebook.

### InceptionV3

To train the model execute:

```bash
$ python3 inceptionv3_train.py
```

To fine tune the model execute:
```bash
$ python3 inceptionv3_finetune.py
```

The model can be test it using the section called "Test the model" in the notebook "notebooks/05 - InceptionV3 pre-trained model.ipynb". The code from inceptionv3_train.py and inceptionv3_finetune.py is included in the notebook but it was not used from the notebook.