import cv2
import os

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter('out_video.mp4', fourcc, 20.0, (960, 540))

images_in_folder = os.listdir('video_output_imgs/')

for i in range(270):
    if str(i)+'.jpg' in images_in_folder:
        frame = cv2.imread('video_output_imgs/'+str(i)+'.jpg')
        out.write(frame)
out.release()
cv2.destroyAllWindows()
