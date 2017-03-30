ORIGINAL_TRAIN_DIRECTORY = "../data/original_train/"
TRAIN_DIRECTORY = "../data/train/"
VALID_DIRECTORY = "../data/valid/"
TEST_DIRECTORY = "../data/test/"

CLASSES = ['cat', 'dog']

VALIDATION_SIZE = 0.2 # size of the validation we want to use
TEST_SIZE = 0.1

import glob
import os
import shutil
import numpy as np

###############
# Folder structure
###############
shutil.rmtree(os.path.join(TEST_DIRECTORY, "dog"), ignore_errors=True)
shutil.rmtree(os.path.join(TEST_DIRECTORY, "cat"), ignore_errors=True)

shutil.rmtree(os.path.join(VALID_DIRECTORY, "dog"), ignore_errors=True)
shutil.rmtree(os.path.join(VALID_DIRECTORY, "cat"), ignore_errors=True)

shutil.rmtree(os.path.join(TRAIN_DIRECTORY, "dog"), ignore_errors=True)
shutil.rmtree(os.path.join(TRAIN_DIRECTORY, "cat"), ignore_errors=True)

os.mkdir(os.path.join(TEST_DIRECTORY, "dog"))
os.mkdir(os.path.join(TEST_DIRECTORY, "cat"))

os.mkdir(os.path.join(VALID_DIRECTORY, "dog"))
os.mkdir(os.path.join(VALID_DIRECTORY, "cat"))

os.mkdir(os.path.join(TRAIN_DIRECTORY, "dog"))
os.mkdir(os.path.join(TRAIN_DIRECTORY, "cat"))

#########################
# DOGS
##########
#random list of dog files
dog_pattern = ORIGINAL_TRAIN_DIRECTORY + "dog.*"
dog_files = np.random.permutation(glob.glob(dog_pattern))

# randomly split the files in train folder and move them to validation
number_validation_dog_files = int(len(dog_files) * VALIDATION_SIZE)
number_test_dog_files = int(len(dog_files) * TEST_SIZE)

for index, dog_file in enumerate(dog_files):
    file_name = os.path.split(dog_file)[1]
    if index < number_validation_dog_files:#validation files
        new_path = os.path.join(VALID_DIRECTORY, "dog",  file_name)
    elif index >= number_validation_dog_files and index < (number_validation_dog_files + number_test_dog_files):
        new_path = os.path.join(TEST_DIRECTORY, "dog",  file_name)
    else:
        new_path = os.path.join(TRAIN_DIRECTORY, "dog",  file_name)
    shutil.copy(dog_file, new_path)

#########################
# CATS
##########
#random list of dog files
cat_pattern = ORIGINAL_TRAIN_DIRECTORY + "cat.*"
cat_files = np.random.permutation(glob.glob(cat_pattern))

# randomly split the files in train folder and move them to validation
number_validation_cat_files = int(len(cat_files) * VALIDATION_SIZE)
number_test_cat_files = int(len(cat_files) * TEST_SIZE)

for index, cat_file in enumerate(cat_files):
    file_name = os.path.split(cat_file)[1]
    if index < number_validation_cat_files:
        new_path = os.path.join(VALID_DIRECTORY, "cat",  file_name)
    elif index >= number_validation_cat_files and index < (number_validation_cat_files+number_test_cat_files):
        new_path = os.path.join(TEST_DIRECTORY, "cat",  file_name)
    else:
        new_path = os.path.join(TRAIN_DIRECTORY, "cat",  file_name)
    shutil.copy(cat_file, new_path)
