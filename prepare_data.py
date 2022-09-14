#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.utils import class_weight
import os, glob, cv2

OLD_SHAPE = (1280, 1920)
NEW_SHAPE = (160, 240)


def padding():

    print("\nPadding... ", end="")

    filenames = np.load(f'data/filenames/filenames.npy')

    for filename in tqdm(filenames, total=len(filenames)):
        # images
        jpg_image = Image.open(f'data/original_images/{filename}.jpg')
        image = np.array(jpg_image)

        padding = image[:, 0, :].copy().reshape(1280, 1, 3).astype(np.uint8)
        new_image = np.concatenate((padding, image), axis=1)

        padding = image[:, -1, :].copy().reshape(1280, 1, 3).astype(np.uint8)
        new_image = np.concatenate((new_image, padding), axis=1)

        jpg_new_image = Image.fromarray(new_image)
        jpg_new_image.save(f'data/padded_images/{filename}.jpg')

        # masks
        jpg_mask = Image.open(f'data/original_masks/{filename}_mask.gif')
        mask = np.array(jpg_mask)

        padding = mask[:, 0].copy().reshape(1280, 1).astype(float)
        new_mask = np.concatenate((padding, mask), axis=1)

        padding = mask[:, -1].copy().reshape(1280, 1).astype(float)
        new_mask = np.concatenate((new_mask, padding), axis=1)

        jpg_new_mask = Image.fromarray(new_mask)
        jpg_new_mask.save(f'data/padded_masks/{filename}_mask.gif')



    
def define_splits(check_bool=False):

    print("\nDefining splits... ", end="")

    # collecting filenames
    filenames = []
    for filename in os.listdir(f'data/padded_images'):
        filename = filename.split(".")[0]
        filenames.append(filename)
    
    filenames = np.array(filenames)

    # splitting filenames
    train_filenames, test_filenames = train_test_split(filenames, test_size=0.3, train_size=0.7, shuffle=True)

    if check_bool: 
        check = len(np.intersect1d(test_filenames, train_filenames))
        if check != 0:
            raise ValueError("Intersection between train set and test set!")
            exit()

    # saving
    np.save(f'data/filenames/filenames', filenames)
    np.save(f'data/filenames/train_filenames', train_filenames)
    np.save(f'data/filenames/test_filenames', test_filenames)

    print('done!')



def resize_data(new_shape=NEW_SHAPE):

    print("\nResizing data...")

    # loading filenames
    train_filenames = np.load(f'data/filenames/train_filenames.npy')
    test_filenames = np.load(f'data/filenames/test_filenames.npy')

    # resizing train set
    print("... for the train set:")
    for train_filename in tqdm(train_filenames, total=len(train_filenames)):
        image = Image.open(f'data/padded_images/{train_filename}.jpg')
        resized_image = image.resize(new_shape[::-1])
        resized_image.save(f'data/train_images/{train_filename}.jpg')

        mask = Image.open(f'data/padded_masks/{train_filename}_mask.gif')
        np_mask = np.array(mask)
        np_resized_mask = cv2.resize(np_mask, new_shape[::-1], interpolation=cv2.INTER_NEAREST)
        resized_mask = Image.fromarray(np_resized_mask.astype(float))
        resized_mask.save(f'data/train_masks/{train_filename}_mask.gif')

    # resizing test set
    print("... for the test set:")
    for test_filename in tqdm(test_filenames, total=len(test_filenames)):
        image = Image.open(f'data/padded_images/{test_filename}.jpg')
        resized_image = image.resize(new_shape[::-1])
        resized_image.save(f'data/test_images/{test_filename}.jpg')

        mask = Image.open(f'data/padded_masks/{test_filename}_mask.gif')
        np_mask = np.array(mask)
        np_resized_mask = cv2.resize(np_mask, new_shape[::-1], interpolation=cv2.INTER_NEAREST)
        resized_mask = Image.fromarray(np_resized_mask.astype(float))
        resized_mask.save(f'data/test_masks/{test_filename}_mask.gif')




def main():

    PADDING = False
    DEFINE_SPLITS = False
    RESIZE_DATA = False

    if PADDING: padding()
    if DEFINE_SPLITS: define_splits()
    if RESIZE_DATA: resize_data()



if __name__ == "__main__":
    main()