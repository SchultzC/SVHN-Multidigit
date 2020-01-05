import vgg_svhn_model as vgg_mod
import custom_svhn_model as cust_mod
import data_helper as dh
import tensorflow as tf
import numpy as np
import h5py
import cv2
import os

# set parameters
which_model = 'custom'
input_folder = 'input_images/'
pyramid_levels = 5
confidence_thresh = 0.9

try:
    # Build the selected model
    if which_model == 'vgg_transfer':
        model = vgg_mod.gen_vgg_16(use_these_weights='vgg_transfer_weights.h5', random_weights=False)
    elif which_model == 'vgg_scratch':
        model = vgg_mod.gen_vgg_16(use_these_weights='vgg_scratch_weights.h5', random_weights=True)
    elif which_model == 'custom':
        model = cust_mod.gen_custom_model(use_these_weights='custom_weights.h5')
    else:
        raise Exception('Please select one of the following:  vgg_transfer,  vgg_scratch,  custom')
except:
    raise Exception('\n\nWARNING!! \n WARNING!!\n'
                    '\nPLEASE ENSURE THAT YOU HAVE DOWNLOADED THE CUSTOM_WEIGHTS.H5 WEIGHTS FILE.  '
                    '\nTHE FILE CAN BE FOUND HERE: \n\n'
                    'https://drive.google.com/open?id=1NetjW2elot0G5L9K-76eKixiWEOFKj-j\n'
                    'PLEASE DOWNLOAD THIS FILE AND PLACE IN THE DIRECTORY CONTAINING run_v3.py')

# get files in input directory
img_files = os.listdir(input_folder)
print('Images to process:\n', img_files)

# loop through the .png files in this directory
for f in img_files:
    if f[-4::] == '.png':
        print('\n\nCurrently working on:  {}'.format(f))
        final_pred = ''
        # import the image
        img = cv2.imread(input_folder+f)

        # create the image pyramid to address multiple scales
        pyr_img = img.copy()
        pyr_imgs = []
        for py_l in range(pyramid_levels):
            pyr_imgs.append(pyr_img)
            pyr_img = cv2.pyrDown(pyr_img)

        # set up the variables for recording the best location
        best_loc = []
        box_size_list = 0
        max_sum = 0
        final_prediciotn = []
        # loop through different levels of the pyramid
        for py_level in range(pyramid_levels):
            print('Pyramid Level :{}:'.format(py_level))
            py_img = pyr_imgs[py_level]
            im_h, im_w, _ = py_img.shape
            for y in range(0, im_h-64, int(16/(2**py_level))):
                for x in range(0, im_w-64, int(16/(2**py_level))):
                    py_imgd = py_img.copy()
                    cut = py_img[y:y+64, x:x+64, :]
                    img_less_mean = dh.less_mean(np.array([cut]).astype(np.float64))

                    prediction = model.predict(img_less_mean, batch_size=1, verbose=0)
                    preds = np.argmax(prediction, axis=2).T
                    pred_list = np.array(preds[0])
                    confidence_list = np.array([prediction[0][0, np.argmax(prediction[0])],
                                                prediction[1][0, np.argmax(prediction[1])],
                                                prediction[2][0, np.argmax(prediction[2])],
                                                prediction[3][0, np.argmax(prediction[3])],
                                                prediction[4][0, np.argmax(prediction[4])]])

                    if np.sum(confidence_list[pred_list != 10]) > max_sum:
                        max_sum = np.sum(confidence_list[pred_list != 10])
                        best_loc = [x*2**py_level, y*2**py_level]
                        box_size = 64*2**py_level
                        final_prediciotn = pred_list[pred_list != 10]

        cv2.rectangle(img, (best_loc[0], best_loc[1]), (best_loc[0]+box_size, best_loc[1]+box_size), (0, 0, 255), 2)
        print(final_prediciotn)
        for numit in final_prediciotn:
            final_pred += str(numit)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, final_pred,  (best_loc[0], best_loc[1]-5), font, 1, (255, 255, 50), 2, cv2.LINE_AA)
        # cv2.imshow(f, img)
        cv2.imwrite('graded_images/OUTPUT_'+f, img)
        # cv2.waitKey(0)
