import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, ImageDataGenerator
import glob
from os.path import join, basename
import os
from shutil import move
import pandas as pd
import sys

#TODO: what preprocessing is necessary? especially what are good values for target_size?

def make_dataloader(filepath='data', target_size=(224, 224), batch_size=100, **kwargs):
    '''Returns a Keras dataloader with specified preprocessing'''
    train_datagen = ImageDataGenerator(**kwargs)
    train_generator = train_datagen.flow_from_directory(filepath,
            target_size=target_size,
            batch_size=batch_size)
    return train_generator

def make_folder_structure(folderpath='data', class_label='style'):
    '''This function assumes that all images are unpacked into folderpath and all_data_info.csv is unpacked there as well.
    class_label indicates which label to use, e.g. artist, style etc (default is style)'''
    image_annot = pd.read_csv(join(folderpath, 'all_data_info.csv'))
    image_annot = image_annot[np.logical_not(image_annot[class_label].isna())]
    # remove pathname so we can easier test if file exists
    filenames = np.array([basename(filename) for filename in glob.glob(join(folderpath, '*.jpg'))])
    # find unique classes we are using, e.g. different styles
    classes = image_annot[class_label].unique()

    for i, class_i in enumerate(classes):
        print('Processing class {0} ({1} of {2} classes).'.format(class_i, i+1, len(classes)))
        # we dont care about nans
        try:
            # make the folder when it doesnt exist yet
            os.mkdir(join(folderpath, class_i))
        except FileExistsError:
            pass
        img_in_this_class = image_annot['new_filename'][image_annot[class_label] == class_i]
        img_that_exist = np.intersect1d(img_in_this_class, filenames)
        for filename in img_that_exist:
           move(join(folderpath, filename), join(folderpath, class_i, filename))

def test_if_folders_are_correct(folderpath, class_label='style'):
    '''Tests if images are in correct folders'''
    image_annot = pd.read_csv(join(folderpath, 'all_data_info.csv'))
    image_annot = image_annot[np.logical_not(image_annot[class_label].isna())]
    # remove pathname so we can easier test if file exists
    filenames = np.array([basename(filename) for filename in glob.glob(join(folderpath, '*.jpg'))])
    pass


if __name__=='__main__':
    if len(sys.argv) > 2:
        folderpath = sys.argv[1:]
        make_folder_struture(folderpath=folderpath)
    else:
        make_folder_structure()

