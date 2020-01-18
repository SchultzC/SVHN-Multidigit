import tensorflow as tf


def gen_custom_model(use_these_weights=None):
    output_categories = 11

    # Set up the input. 64x64
    # Images in the data sets.
    initialization = tf.keras.initializers.glorot_uniform()
    model_input = tf.keras.Input(shape=(64, 64, 3), name='image_64_64_3')

    # Hidden Layer 1
    conv1 = tf.keras.layers.Conv2D(filters=48, kernel_size=[5, 5], kernel_initializer=initialization,
                                   padding='same', activation='relu', name="conv1")(model_input)
    batch_n1 = tf.keras.layers.BatchNormalization(name="batch_n1")(conv1)
    max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same', name="max_pool1")(batch_n1)
    drop_1 = tf.keras.layers.Dropout(0.2, name="drop_1")(max_pool1)

    # Hidden Layer 2
    conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], kernel_initializer=initialization,
                                   padding='same', activation='relu', name='conv2')(drop_1)
    batch_n2 = tf.keras.layers.BatchNormalization(name="batch_n2")(conv2)
    max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=1, padding='same', name="max_pool2")(batch_n2)
    drop_2 = tf.keras.layers.Dropout(0.2, name="drop_2")(max_pool2)

    # Hidden Layer 3
    conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=[5, 5], kernel_initializer=initialization,
                                   padding='same', activation='relu', name='conv3')(drop_2)
    batch_n3 = tf.keras.layers.BatchNormalization(name="batch_n3")(conv3)
    max_pool3 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same', name="max_pool3")(batch_n3)
    drop_3 = tf.keras.layers.Dropout(0.2, name="drop_3")(max_pool3)

    # Hidden Layer 4
    conv4 = tf.keras.layers.Conv2D(filters=160, kernel_size=[5, 5], kernel_initializer=initialization,
                                   padding='same', activation='relu', name='conv4')(drop_3)
    batch_n4 = tf.keras.layers.BatchNormalization(name="batch_n4")(conv4)
    max_pool4 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=1, padding='same', name="max_pool4")(batch_n4)
    drop_4 = tf.keras.layers.Dropout(0.2, name="drop_4")(max_pool4)

    # Hidden Layer 5
    conv5 = tf.keras.layers.Conv2D(filters=192, kernel_size=[5, 5], kernel_initializer=initialization,
                                   padding='same', activation='relu', name='conv5')(drop_4)
    batch_n5 = tf.keras.layers.BatchNormalization(name="batch_n5")(conv5)
    max_pool5 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same', name="max_pool5")(batch_n5)
    drop_5 = tf.keras.layers.Dropout(0.2, name="drop_5")(max_pool5)

    # Hidden Layer 6
    conv6 = tf.keras.layers.Conv2D(filters=192, kernel_size=[5, 5], kernel_initializer=initialization,
                                   padding='same', activation='relu', name='conv6')(drop_5)
    batch_n6 = tf.keras.layers.BatchNormalization(name="batch_n6")(conv6)
    max_pool6 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=1, padding='same', name="max_pool6")(batch_n6)
    drop_6 = tf.keras.layers.Dropout(0.2, name="drop_6")(max_pool6)

    # Hidden Layer 7
    conv7 = tf.keras.layers.Conv2D(filters=192, kernel_size=[5, 5], kernel_initializer=initialization,
                                   padding='same', activation='relu', name='conv7')(drop_6)
    batch_n7 = tf.keras.layers.BatchNormalization(name="batch_n7")(conv7)
    max_pool7 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding='same', name="max_pool7")(batch_n7)
    drop_7 = tf.keras.layers.Dropout(0.2, name="drop_7")(max_pool7)

    # Hidden Layer 8
    conv8 = tf.keras.layers.Conv2D(filters=192, kernel_size=[5, 5], kernel_initializer=initialization,
                                   padding='same', activation='relu', name='conv8')(drop_7)
    batch_n8 = tf.keras.layers.BatchNormalization(name="batch_n8")(conv8)
    max_pool8 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=1, padding='same', name="max_pool8")(batch_n8)
    drop_8 = tf.keras.layers.Dropout(0.2, name="drop_8")(max_pool8)

    flatten_convs_output = tf.keras.layers.Flatten(name="flatten_conv_output")(drop_8)

    # Hidden Layer 9
    fc_1 = tf.keras.layers.Dense(units=3072, activation="relu", name="fc_1")(flatten_convs_output)

    # Hidden Layer 10
    fc_2 = tf.keras.layers.Dense(units=3072, activation="relu", name="fc_2")(fc_1)

    # Output_layer digits
    output_d1 = tf.keras.layers.Dense(units=output_categories, activation="softmax", name="digit1")(fc_2)
    output_d2 = tf.keras.layers.Dense(units=output_categories, activation="softmax", name="digit2")(fc_2)
    output_d3 = tf.keras.layers.Dense(units=output_categories, activation="softmax", name="digit3")(fc_2)
    output_d4 = tf.keras.layers.Dense(units=output_categories, activation="softmax", name="digit4")(fc_2)
    output_d5 = tf.keras.layers.Dense(units=output_categories, activation="softmax", name="digit5")(fc_2)

    # now create an output array that matches our data labels
    final_output = [output_d1, output_d2, output_d3, output_d4, output_d5]
    custom_model = tf.keras.Model(inputs=model_input, outputs=final_output)

    # We want to make sure that if we are running this model during testing
    # that we can return just the trained model without retraining it.
    if use_these_weights:
        custom_model.load_weights(use_these_weights)
    custom_model.summary()

    return custom_model
