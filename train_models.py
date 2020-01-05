import vgg_svhn_model as vgg_mod
import custom_svhn_model as cust_mod
import data_helper as dh
import tensorflow as tf
import numpy as np
import h5py

clean_data_only = True
which_model = 'custom'  # CHOOSE ONE:  'vgg_scratch', 'vgg_transfer', 'custom'

weights_file = which_model+"_weights.h5"

selected_optimizer = tf.keras.optimizers.Nadam()

# Lets get the Data:
if clean_data_only:
    h5f = h5py.File('clean_training_images.h5', 'r')
else:
    h5f = h5py.File('all_training_images.h5', 'r')

training_images = h5f['training_images'][:]
training_labels = h5f['training_labels'][:]
validation_images = h5f['validation_images'][:]
validation_labels = h5f['validation_labels'][:]
h5f.close()
print('----Data has been Imported----')


# The data as it currently stands is ordered.  This is because we added to the data from multiple sources
# including 'train', 'extra' and 'no_number' data.  In its current configuration all of the no_number
# data is at the end.  To avoiding making batches of data in training that are all of one type (i.e. all no_numbers)
# we will shuffle the data

# get the new order
training_shuffle_order = dh.shuffle_data_order(training_labels.shape[0])
validation_shuffle_order = dh.shuffle_data_order(validation_labels.shape[0])
# perform shuffling on data set.
training_images = training_images[training_shuffle_order]
training_labels = training_labels[training_shuffle_order]
validation_images = validation_images[validation_shuffle_order]
validation_labels = validation_labels[validation_shuffle_order]


# We should ensure that the labels match known model output shape
train_labels_trans = np.copy(training_labels).transpose()
train_labels_final = [train_labels_trans[0], train_labels_trans[1], train_labels_trans[2],
                      train_labels_trans[3], train_labels_trans[4]]

validation_labels_trans = np.copy(validation_labels).transpose()
validation_labels_final = [validation_labels_trans[0], validation_labels_trans[1], validation_labels_trans[2],
                           validation_labels_trans[3], validation_labels_trans[4]]

# Now as a pre-processing step we should subtract the mean from each image
training_images_less_mean = dh.less_mean(training_images.astype(np.float64))
validation_images_less_mean = dh.less_mean(validation_images.astype(np.float64))

# Build the selected model
if which_model == 'vgg_transfer':
    model = vgg_mod.gen_vgg_16(use_these_weights=None, random_weights=False)
elif which_model == 'vgg_scratch':
    model = vgg_mod.gen_vgg_16(use_these_weights=None, random_weights=True)
elif which_model == 'custom':
    model = cust_mod.gen_custom_model(use_these_weights=None)
else:
    raise Exception('Please select one of the following:  vgg_transfer,  vgg_scratch,  custom')


model.compile(loss='sparse_categorical_crossentropy', optimizer=selected_optimizer, metrics=['accuracy'])

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=weights_file, monitor='val_loss',
                                                verbose=1, save_best_only=True),

             tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto'),

             tf.keras.callbacks.TensorBoard(log_dir='checkpoints/'+which_model, histogram_freq=0,
                                            write_graph=True, write_images=True)]

training_stats = model.fit(x=training_images_less_mean, y=train_labels_final,
                           validation_data=(validation_images_less_mean, validation_labels_final),
                           epochs=3, batch_size=64, verbose=1, callbacks=callbacks)

# Save
model.save_weights(weights_file)
model.save(which_model + '.h5')

