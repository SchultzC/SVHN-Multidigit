import data_helper as dath
import h5py
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

#path to no_number data to be processed.
path_to_dir = 'raw_data/no_numbers/temp/'
wanted_size = 64

paths_to_image_files = tf.gfile.Glob(os.path.join(path_to_dir, '*.png'))

counter = 1
for img_path in paths_to_image_files:
    img = cv2.imread(img_path)
    img_height, img_width, img_chan = img.shape

    if counter % 100 == 0:
        print(counter, img.shape)

    # if its a large image in both dimensions, split into 4. If only one direction is large
    # split only on that direction, otherwise don't change it but copy to the right location.

    if(img_height > wanted_size * 2) and (img_width > wanted_size * 2):
        img_1 = img[0:int(img_height/2), 0:int(img_width/2), :]
        img_2 = img[0:int(img_height/2), int(img_width/2)::, :]
        img_3 = img[int(img_height/2)::, 0:int(img_width/2), :]
        img_4 = img[int(img_height/2)::, int(img_width/2)::, :]

        cv2.imwrite('raw_data/no_numbers/'+str(counter)+'.png', img_1)
        counter += 1
        cv2.imwrite('raw_data/no_numbers/' + str(counter) + '.png', img_2)
        counter += 1
        cv2.imwrite('raw_data/no_numbers/' + str(counter) + '.png', img_3)
        counter += 1
        cv2.imwrite('raw_data/no_numbers/' + str(counter) + '.png', img_4)
        counter += 1

    elif img_height > wanted_size * 2:
        img_1 = img[0:int(img_height / 2), :, :]
        img_2 = img[int(img_height / 2)::, :, :]
        cv2.imwrite('raw_data/no_numbers/' + str(counter) + '.png', img_1)
        counter += 1
        cv2.imwrite('raw_data/no_numbers/' + str(counter) + '.png', img_2)
        counter += 1

    elif img_width > wanted_size * 2:
        img_1 = img[:, 0:int(img_width / 2), :]
        img_2 = img[:, int(img_width / 2)::, :]
        cv2.imwrite('raw_data/no_numbers/' + str(counter) + '.png', img_1)
        counter += 1
        cv2.imwrite('raw_data/no_numbers/' + str(counter) + '.png', img_2)
        counter += 1

    else:
        cv2.imwrite('raw_data/no_numbers/' + str(counter) + '.png', img)
        counter += 1

    # this is a final set for flipping the reduced photos

    img_flip = cv2.flip(img, 1)

    # cv2.imwrite('raw_data/no_numbers/' + str(counter) + '.png', img)
    counter += 1
    # cv2.imwrite('raw_data/no_numbers/' + str(counter) + '.png', img_flip)
    counter += 1
