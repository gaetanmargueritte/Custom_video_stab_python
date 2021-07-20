# not oriented FAST algorithm: applies orientation to FAST detected keypoints, home-made by Gaëtan MARGUERITTE.
import numpy as np
import matplotlib.pyplot as plt
import cv2
from nFAST import nFAST
from math import atan2

def find_moments(patch):
    y, x = np.mgrid[:patch.shape[0], :patch.shape[1]] 
    m00 = np.sum(patch) # sum of all intensities in the patch
    m10 = np.sum(x * patch) # regarding of x
    m01 = np.sum(y * patch) # regarding of x
    return [m00, m10, m01]

def find_centroid(moments):
    return [moments[1]/moments[0], moments[2]/moments[0]]

def find_orientations(img, fps, radius):
    indices = np.nonzero(fps)
    orientations = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=float)
    index = 0
    for i in indices[0]:
        j = indices[1][index]
        patch = img[i-radius:i+radius+1, j-radius:j+radius+1]
        moments = find_moments(patch)
        orientations[i, j] = atan2(moments[2], moments[1])
        index = index + 1
    return orientations
