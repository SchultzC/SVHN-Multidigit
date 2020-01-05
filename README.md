# SVHN-Multidigit - Experimenting with CNNs and the SVHN dataset
Ian Goodfellow and collaborators in 2014, achieved state of the art (at the time) single digit accuracy (97.8%) while performing classification on up to 6 digits simultaneously using SVHN.  This approach required a modification in the way that dataset is typically preprocessed.  I demonstrate that if the dataset is used in this way, there will be errors in the new preprocessed dataset that were inconsequential previously (in single digit use case).  Using a simple script I provide a method for cleaning the training dataset so as to remove images that would cause these errors in the the new preprocessing method.  The result is an improvement in training and test accuracy when comparing minimally trained networks based on the the models in the original work.

In addition I show the ability to use these models for detecting (up to) six digit house numbers in video.


### The is a project done with python 3 and requires the following: 
- Tensorflow 
- CV2 
- Numpy 
- H5PY


### Here is a brief description for the use for each python file:

-  run_v3.py:  This can be run by simply running the following command within a python3 
environment:       python run_v3.py

-  data_helper.py:   This contains a few helpful functions used in various other scripts

-  svhn_view_and_bad_data.py:  recognizes bad data for the purpose of sequence 
classification

-  NoNumberDataMakers.py:  This does the data augmentation to create

-  NoNumber_expand.py:  This takes the no-digit data and processes it to create up to 
40K no digit data points

-  custom_svhn_model.py:  This will create custom model as described in the report

-  vgg_svhn_modl.py:  This will create the vgg model as described in the report

-  train_models.py:   This will train the above models as selected by the user

-  test_models.py:   This will test the selected models on the SVHN test set.

-  use_model_video.py:   This will run classification and detection using a selected model
on an input video

-   make_video.py:    This simply reads in a directory of processed video frames images
and creates the video in .mp4 format


Other:
As mentioned in the write up, data cleaning was done (training data only) to achieve better 
results.  A list of the images that were removed and the reason they were removed 
is contained in:                  removed_data.csv