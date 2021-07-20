import numpy as np # array management
import math        # math functions
import cv2         # image management

"""
returns a gaussian filter of given size
"""
def gauss(size, sigma):
    filter_1d = np.linspace(-size/2, size/2, size)
    for i in range(size):
        val = filter_1d[i]
        filter_1d[i] =  1 / (np.sqrt(2 * np.pi) * sigma) * np.e ** (-np.power(val / sigma, 2) / 2)
    filter_2d = np.outer(filter_1d.T, filter_1d.T)
    filter_2d = 1. / filter_2d.max()
    return filter_2d

def apply_filter(img, gauss_filter, size):
    rows, cols = img.shape
    output = np.zeros((rows, cols))
    # padded image to apply filter on borders of the picture
    padded = np.zeros((rows + size, cols + size))
    padded[size:rows+size, size:cols + size] = img
    for i in range(rows):
        for j in range(cols):
            output[i, j] = np.sum(gauss_filter * padded[i:i + size, j:j + size]) / (size * size)
    return output
    
    

"""
applies a gaussian blur of default size 5 on input image
returns the filtered image
"""
def gaussian_blur(img, size=5):
    gauss_filter = gauss(size, math.sqrt(size))
    return apply_filter(img, gauss_filter, size) 
