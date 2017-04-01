from __future__ import division,print_function

import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numbers
from math import sqrt
from math import ceil
from sklearn.utils import shuffle  #replace this
from scipy import ndimage
from scipy import misc

import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


def read_dataset_from_directory(directory):
    '''
    Read all the images and labels from the directory. It use a fixed directory structure, where images from each category must be placed in a separate folder. (For example: DOG/dog_01.jpg)
    
    Read a directory with the structure CLASS/img_ and return a pandas dataframe with the path and the class for each image.
    
    Args:
        directory (string): Directory to list
    '''
    class_folders = glob.glob(directory + '/*')
    files = [glob.glob(cls + '/*') for cls in class_folders] # put class info with file name
    files = [img for cls in files for img in cls]
    
    df = pd.DataFrame({'fpath':files,'w':0,'h':0})

    df['category'] = df.fpath.str.extract('([a-zA-Z]*)/img_', expand=False) # extract class
    
    for idx in df.index:
        im = Image.open(df.ix[idx].fpath)
        df.ix[idx,['w','h']] = im.size
    
    
    return df


def get_smaller_grid_size(number_images):
    '''
    Calculates the smallest grid size for any given number of images.
    
    Args:
        number_images (int): number of images
        
    Return:
        (image batch, label batch): Image and label batch
    '''
    assert number_images > 0
    
    out = ceil(sqrt(number_images))
    return  out, out


def plot_grid(ims, interp=False, titles=None):
    
    '''
    
    Given an array with images 'ims', this function plot every image in a grid.

    Args:
        ims (array): array of images
        interp (string): interpolation method, see: http://matplotlib.org/1.4.3/examples/images_contours_and_fields/interpolation_methods.html
        titles (string): Titles for the plots (eg.: the classes of each image)
        
    Return:
        (image batch, label batch): Image and label batch
        
    
    '''
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3): # is not a channel last image?
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=(15,10))
    f.subplots_adjust(wspace=0.02,hspace=0)
    
    width, height = get_smaller_grid_size(len(ims))
    for i in range(len(ims)):
        sp = f.add_subplot(width, height, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.title.set_text(titles[i])
        plt.imshow(ims[i], interpolation=None if interp else 'none')

def get_keras_batch_generator(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',
                target_size=(224,224)):
    '''
    
    Returns a generator that produces batches of features (images) and labels (classes) from a given directory. The directory should have a fixed structure, CATEGORY/img_1.jpg.

    Args:
        dirname (string): path of the data directory
        gen (generator): generator to be used, keras ImageDataGenerator used by default
        shuffle (string): Titles for the plots (eg.: the classes of each image)
        batch_size (int): batch size
        class_mode (string): mode for the classes 
        target_size (int, int): size of the images to returned
        
    Return:
        (imgs, label): batch of data
        
    Examples:
        get_batches(path+'valid', batch_size=batch_size*2, shuffle=False)
    
    '''    
    
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)