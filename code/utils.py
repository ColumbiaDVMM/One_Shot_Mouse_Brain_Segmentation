from __future__ import print_function
import cv2
import numpy as np
import os
import shutil
from PIL import Image, ImageEnhance
import sklearn.neighbors

color = [(101, 85, 237), (104, 212, 160), (233, 193, 79),
         (84, 206, 255), (173, 207, 72), (127, 127, 127)]  # (0, 0, 0),

# draw image with multi-channel mask


def drawMultiRegionMultiChannel(mask):
    final_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # print(mask.shape)

    for i in range(mask.shape[2]):
        if i == 0:
            final_image[mask[:, :, i] > 100, :] = (0, 0, 0)
        else:
            final_image[mask[:, :, i] > 100, :] = color[i % len(color)]

    return final_image

# draw image with index mask


def drawMultiRegionIndex(index_mask):
    final_image = np.zeros(
        (index_mask.shape[0],
         index_mask.shape[1],
         3),
        dtype=np.uint8)

    for i in range(np.max(index_mask) + 1):
        if i == 0:
            final_image[index_mask == i, :] = (0, 0, 0)
        else:
            final_image[index_mask == i, :] = color[i % len(color)]

    return final_image


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Clear and remake directory
def ClearDirectory(*args):
    if args:
        for path in args:
            if os.path.exists(path):
                shutil.rmtree(path)
                os.makedirs(path)

#%% pre-process reference mask with edge as reference
#  Resize and crop to realize expand
def expandImage(im, expand_pixel=30):
    expand_img = im
    edge = cv2.Canny(expand_img, 50, 500)
    width, height = edge.shape

    edge_list = []
    for i in range(width):
        for j in range(height):
            if edge[i, j] >= 100:
                edge_list.append((i, j))

    edge_tree = sklearn.neighbors.KDTree(edge_list)

    for i in range(width):
        for j in range(height):
            dist, ind = edge_tree.query([(i, j)], k=1)
            if expand_img[i, j] < 100 and dist < expand_pixel:
                expand_img[i, j] = 255

    # for the test
    # cv2.imwrite('../data/tem/expand_mask.png', expand_img)
    return expand_img

def assignWeight2Mask(ref_mask):
    edge = cv2.Canny(ref_mask, 50, 500)
    width, height = ref_mask.shape

    # extract central region without edge
    ref_mask_no_edge = ref_mask - edge

    # get coordinates of all edge pixels
    edge_list = []
    for i in range(width):
        for j in range(height):
            if edge[i, j] >= 100:
                edge[i, j] = 127.5
                edge_list.append((i, j))
    
    edge_tree = sklearn.neighbors.KDTree(edge_list)

    new_ref_mask = ref_mask_no_edge + edge
    
    assign_scale = (np.log(99) - np.log(1.0/99))/(width/2)
    print('assign scale:', assign_scale)

    # loop over all pixels in the central area and compute it's distance to it's nearest edge
    for i in range(width):
        for j in range(height):
            dist, ind = edge_tree.query([(i, j)], k=1)
            if new_ref_mask[i, j] == 255:
                ref_mask_no_edge[i, j] = sigmoid(dist * assign_scale) * 255
            elif new_ref_mask[i, j] == 0:
                ref_mask_no_edge[i, j] = sigmoid((-dist) * assign_scale) * 255

    # merge central area with edge
    ref_mask_new = ref_mask_no_edge + edge
    return ref_mask_new
