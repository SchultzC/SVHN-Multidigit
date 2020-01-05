import h5py
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2


def get_data(digitstruct_file, index):
    data_array = {}
    item = digitstruct_file['digitStruct']['bbox'][index].item()

    for key in ['label', 'left', 'top', 'width', 'height']:
        data_feature = digitstruct_file[item][key]

        values = [digitstruct_file[data_feature.value[k].item()].value[0][0]
                  for k in range(len(data_feature))] if len(data_feature) > 1 else \
            [data_feature.value[0][0]]

        data_array[key] = values
    return data_array

only_clean = True
data_file = 'extra'
path_to_dir = 'raw_data/' + data_file
wanted_size = 64

exclusion_list = []
if only_clean:
    exclusion_list = pd.read_csv('raw_data/bad_' + data_file + '_data.csv').ix[:, 0].tolist()


path_digitstruct_mat_file = os.path.join(path_to_dir, 'digitStruct.mat')
path_to_image_files = tf.gfile.Glob(os.path.join(path_to_dir, '*.png'))
total_files = len(path_to_image_files)
print("Total Number of Files in the given directory: {}".format(total_files))
count = 15190
for i in range(1, total_files+1):
    if (i not in exclusion_list) or (exclusion_list is None):
        path_to_image_file = os.path.join(path_to_dir, str(i) + '.png')
        index_loc = int(path_to_image_file.split('/')[-1].split('.')[0]) - 1

        with h5py.File(path_digitstruct_mat_file, 'r') as digitstruct_mat_file:
            this_data = get_data(digitstruct_mat_file, index_loc)
            img = cv2.imread(path_to_image_file)
            im_height, im_width, im_chans = img.shape
            left_margin = this_data['left'][0]
            right_margin = im_width - (this_data['left'][-1] + this_data['width'][-1])

            crop_left = None
            crop_right = None
            if (left_margin > wanted_size) and (im_height > wanted_size):
                crop_left = img[:, 0:int(this_data['left'][0]), :]

            if (right_margin > wanted_size) and (im_height > wanted_size):
                crop_right = img[:, int(this_data['left'][-1] + this_data['width'][-1])::, :]

            if (crop_left is not None) & (crop_right is not None):

                if np.random.random() > 0.5:
                    desired_crop = crop_right
                else:
                    desired_crop = crop_left

            elif crop_left is not None:
                desired_crop = crop_left

            elif crop_right is not None:
                desired_crop = crop_right

            else:
                desired_crop = None

            if desired_crop is not None:
                count += 1
                cv2.imwrite('raw_data/no_numbers/' + str(count) + '.png', desired_crop)
                if count % 100 == 0:
                    print(count, i)
                # cv2.imshow('s', desired_crop)
                # cv2.waitKey(0)


