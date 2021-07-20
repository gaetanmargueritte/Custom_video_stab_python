# notORB algorithm: mimicks the behavior of ORB algorithm, home-made by Gaëtan MARGUERITTE
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import math
import scipy.spatial.distance as scp

from FeaturePoint import FeaturePoint
from scipy import ndimage
from nFAST import nFAST
from nANMS import nANMS
from nOFAST import find_orientations
from gaussian import gaussian_blur
from nBRIEF import generate_samples, nSBRIEF

t = 100
feature_radius = 3
vector_size = 256

"""
gets nb feature points from the image img
"""
def get_feature_points(img, max_fps, verbose=False):
    start = time.time()
    feature_points = nFAST(img, t)
    end = time.time()

    if verbose:
        print("worktime to find interest points through nFAST = ", end-start)
        print("image shape (rows, cols) = ", img.shape)
        print("feature radius =", feature_radius)
        print("feature threshold =", t)
        print("number of feature points detected =", len(np.nonzero(feature_points)[0]))

    start = time.time()
    # starting suppression radius of ANSM set at 10 times the feature point size
    nanms_feature_points = nANMS(feature_points, shape=img.shape, nb=max_fps, radius=feature_radius)
    end = time.time()

    if verbose:
        print("worktime to use nANMS = ", end-start)
        print("number of feature points detected = ", len(np.nonzero(nanms_feature_points)[0]))
    return nanms_feature_points

"""
gets descriptor list of the input image
"""
def get_descriptors(img, fps, samples, verbose=False):
    start = time.time()
    indices = np.nonzero(fps) 
    result = np.empty(shape=(len(indices[0])), dtype=object)
    s = 2*feature_radius
    orientations = find_orientations(img, fps, feature_radius)
    index = 0
    for i in indices[0]:
        j = indices[1][index]
        if 0 <= i-s and  i+s+1 < img.shape[0] and 0 <= j-s and  j+s+1 < img.shape[1]:   
            # create a patch of uneven size S, centered around the feature point
            patch = img[i-s:i+s+1, j-s:j+s+1]
            # removes noises that would disturb rbrief descriptor
            blurred_patch = gaussian_blur(patch, s*2+1)
            descriptor = nSBRIEF(blurred_patch, s, vector_size, samples, orientations[i, j])
        else:
            descriptor = np.array([False for i in range(vector_size)])
        result[index] = FeaturePoint([i, j], descriptor)
        index = index + 1
    end = time.time()

    if verbose:
        print("worktime to find descriptor of each patch = ", end-start)

    return result
    
def nBFM(fps, fps2, top, verbose=False):
    start = time.time()
    descriptors = np.array([f.descriptor for f in fps])
    descriptors2 = np.array([f.descriptor for f in fps2])
    # pairwise distance of each descriptor
    distances = scp.cdist(descriptors, descriptors2, metric='hamming')

    # index of point in fps2 that is the closest to a point in fps
    matches = np.argmin(distances, axis=1)
    # minimal distance for each point in fps
    dist = np.array([distances[i, matches[i]] for i in range(len(fps))])
    sorted_dist = np.argsort(dist)    
    
    # we want the number 'top' closest matches
    indices = np.zeros(shape=(top, 2), dtype=int)
    for i in range(top):
        indices[i][0] = sorted_dist[i]
        indices[i][1] = matches[sorted_dist[i]]
    end = time.time()

    if verbose:
        print(top, " selected best matches keypoints")
        print("worktime to match descriptors of two images = ", end-start)

    return indices


"""
notORB: receives two frames in grayscale and find corresponding keypoints between them
returns a FeaturePoint list of matches
"""
def nORB(img, img2, verbose=False, features_img1=None):
    # number of maximum feature points searched
    max_fps = 400

    if features_img1 is None:
        feature_points = get_feature_points(img, max_fps, True)
    else: # simply regenerate matrix
        feature_points = np.zeros(shape=img.shape, dtype=int)
        for i in range(len(features_img1)):
            position = features_img1[i].position
            feature_points[position[0], position[1]] = img[position[0], position[1]]
    feature_points2 = get_feature_points(img2, max_fps, True)

    s = 6
    vector_size = 256
    # generate samples to test two pictures
    samples = generate_samples(s, vector_size)
    
    # get descriptor for each normalized patch
    fps = get_descriptors(img, feature_points, samples, True)
    fps2 = get_descriptors(img2, feature_points2, samples, True)

    
    top = min(len(fps), len(fps), 75)
    matches = nBFM(fps, fps2, top, True)

    return (fps, fps2, matches)
    
