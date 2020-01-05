import vgg_svhn_model as vgg_mod
import custom_svhn_model as cust_mod
import data_helper as dh
import tensorflow as tf
import numpy as np
import h5py

which_model = 'custom'  # CHOOSE ONE:  'vgg_scratch', 'vgg_transfer', 'custom'

h5f = h5py.File('clean_testing_images.h5', 'r')

selected_optimizer = tf.keras.optimizers.Nadam()

testing_images = h5f['testing_images'][:]
testing_labels = h5f['testing_labels'][:]
h5f.close()
print('----Data has been Imported----')

# We should ensure that the labels match known model output shape
testing_labels_trans = np.copy(testing_labels).transpose()
testing_labels_final = [testing_labels_trans[0], testing_labels_trans[1], testing_labels_trans[2],
                        testing_labels_trans[3], testing_labels_trans[4]]

# Now as a pre-processing step we should subtract the mean from each image
testing_images_less_mean = dh.less_mean(testing_images.astype(np.float64))

# Build the selected model
if which_model == 'vgg_transfer':
    model = vgg_mod.gen_vgg_16(use_these_weights='vgg_transfer_weights.h5', random_weights=False)
elif which_model == 'vgg_scratch':
    model = vgg_mod.gen_vgg_16(use_these_weights='vgg_scratch_weights.h5', random_weights=True)
elif which_model == 'custom':
    model = cust_mod.gen_custom_model(use_these_weights='custom_weights.h5')
else:
    raise Exception('Please select one of the following:  vgg_transfer,  vgg_scratch,  custom')

model.compile(loss='sparse_categorical_crossentropy', optimizer=selected_optimizer, metrics=['accuracy'])
results = model.evaluate(x=testing_images_less_mean, y=testing_labels_final, batch_size=64, verbose=1)
print(results)
