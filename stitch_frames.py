import cv2
import numpy as np
import os

# Adapted from https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
def get_files(path, int_sort = False):
    included_extensions = ['jpg','jpeg', 'bmp', 'png', 'gif']
    lst = os.listdir(path)
    if int_sort:
        lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    file_names = [fn for fn in lst
        if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

path = 'pipeline/out_frames/mtcnn_confident/'
lst = get_files(path, True)

# Get all frames
img_array = []
for i in range(len(lst)):
    frame = lst[i]
    img = cv2.imread(path + frame)
    height, width, channels = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()