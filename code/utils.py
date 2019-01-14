from __future__ import print_function
import cv2
import numpy as np
import glob
import os
import shutil
from PIL import Image, ImageEnhance
from collections import namedtuple
from operator import itemgetter
from pprint import pformat
import sklearn.neighbors 

color = [(101, 85, 237), (104, 212, 160), (233, 193, 79),\
        (84, 206, 255), (173, 207, 72), (127, 127, 127)]#(0, 0, 0), 

#draw image with multi-channel mask
def drawMultiRegionMultiChannel(mask):
    final_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype = np.uint8)
    #print(mask.shape)

    for i in range(mask.shape[2]):
        if i == 0:
            final_image[mask[:,:,i]>100,:] = (0,0,0)
        else:
            final_image[mask[:,:,i]>100,:] = color[i%len(color)]

    return final_image

#draw image with index mask
def drawMultiRegionIndex(index_mask):
    final_image = np.zeros((index_mask.shape[0], index_mask.shape[1], 3), dtype = np.uint8)

    for i in range(np.max(index_mask)+1):
        if i == 0:
            final_image[index_mask==i,:] = (0,0,0)
        else:
            final_image[index_mask==i,:] = color[i%len(color)]

    return final_image


def sigmoid(x):
    return 1/(1 + np.exp(-x))


# Clear and remake directory
def ClearDirectory(*args):
    if args:
        for path in args:
            if os.path.exists(path):
                shutil.rmtree(path)
                os.makedirs(path)
