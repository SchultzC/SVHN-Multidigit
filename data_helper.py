import numpy as np
# import tensorflow as tf


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


def data_location_choice(split_perc):
    if np.random.random() > split_perc[1]:
        return 0
    else:
        return 1


def shuffle_data_order(data_len):
    p = np.random.permutation(data_len)
    return p


def less_mean(data_set_images):
    # print('starting mean subtraction for images')
    for img in range(data_set_images.shape[0]):
        data_set_images[img] -= data_set_images[img].mean()
    # print('mean subtraction complete')
    return data_set_images

# artifact from trying previously to tfRecords with Keras. Oops.
# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

