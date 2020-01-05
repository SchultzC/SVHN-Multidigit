import tensorflow as tf


def gen_vgg_16(use_these_weights=None, random_weights=False):
    """Get the VGG model from tf.keras.
    I'm choosing to not include the 3 fully-connected layers
    at the top of the network because I will be changing both
    the input and output size. I want the network to have hidden
    layers that can take advantage pre-trained VGG weights without
    being overly constrained by them."""

    if not random_weights:
        vgg_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    else:
        vgg_model = tf.keras.applications.VGG16(include_top=False, weights=None)

    # Set up the input. 64x64
    # Images in the data sets.
    model_input = tf.keras.Input(shape=(64, 64, 3), name='image_64_64_3')

    # These will represent 0-9 and 10 for 'no digit'
    output_categories = 11

    # Get an output from the portion of the vgg network that we kept
    vgg_out = vgg_model(model_input)

    # Since we removed the top three FC layers we should add them back on top.
    # We will be moving towards our the input that will be recieved by each of the
    # digit classifier outputs

    # First we will flatten vgg_out
    flat_out = tf.keras.layers.Flatten(name='vgg_out_flat')(vgg_out)

    # In Vgg the fully connected layers at the end are of size 4096
    # coming from a 256x256 input image (16 X 256) since our input will be
    # 64x64 our fully connected layers will have size 1024 (16 X 64)
    fc_1 = tf.keras.layers.Dense(1024, activation='relu', name='fc_1')(flat_out)
    fc_2 = tf.keras.layers.Dense(1024, activation='relu', name='fc_2')(fc_1)
    fc_3 = tf.keras.layers.Dense(1024, activation='relu', name='fc_3')(fc_2)

    # Now the weird part,  because we are classifying all the digits at once
    # we need to pass the fc_3 to 5 different output layers each of which
    # will predict a number 0-9 or 10 for 'no digit'
    output_d1 = tf.keras.layers.Dense(units=output_categories, activation="softmax", name="digit1")(fc_3)
    output_d2 = tf.keras.layers.Dense(units=output_categories, activation="softmax", name="digit2")(fc_3)
    output_d3 = tf.keras.layers.Dense(units=output_categories, activation="softmax", name="digit3")(fc_3)
    output_d4 = tf.keras.layers.Dense(units=output_categories, activation="softmax", name="digit4")(fc_3)
    output_d5 = tf.keras.layers.Dense(units=output_categories, activation="softmax", name="digit5")(fc_3)

    # now create an output array that matches our data labels
    final_output = [output_d1, output_d2, output_d3, output_d4, output_d5]

    new_vgg_model = tf.keras.Model(inputs=model_input, outputs=final_output)

    # We want to make sure that if we are running this model during testing
    # that we can return just the trained model without retraining it.
    if use_these_weights:
        new_vgg_model.load_weights(use_these_weights)
    new_vgg_model.summary()

    return new_vgg_model
