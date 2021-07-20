# notBRIEF algorithm: mimicks the behavior of BRIEF algorithm, home-made by Gaëtan MARGUERITTE
# Scale invariance has not been added to previous FAST steps, only rotational invariance has been added
# used in the nORB (notORB) algorithm
import numpy as np
import cv2
import random
import math

"""
Generates sample points, according to method Gaussian II in the litterature provided by BRIEF: Binary Robust Independent Elementary Features paper (Michael Calonder, Vincent Lepetit, Christoph Strecha, and Pascal Fua)
"""
def generate_samples(patch_radius, vector_size):
    patch_size = (patch_radius*2) + 1
    random.seed()
    samples = np.zeros(shape=(vector_size, 4), dtype=int)
    for i in range(vector_size):
        p1r = p1c = p2r = p2c = -patch_radius-1
        while not(-patch_radius < p1r < patch_radius):
            p1r = round(random.gauss(0, 0.04 * (patch_size ** 2)))
        while not(-patch_radius < p1c < patch_radius):
            p1c = round(random.gauss(0, 0.04 * (patch_size ** 2)))
        while not(-patch_radius < p2r < patch_radius):
            p2r = round(random.gauss(0, 0.04 * (patch_size ** 2)))
        while not(-patch_radius < p2c < patch_radius):
            p2c = round(random.gauss(0, 0.04 * (patch_size ** 2)))
        samples[i] = [int(p1r), int(p1c), int(p2r), int(p2c)]
    return samples

"""
A short nBRIEF descriptor, receiving a smoothed orientation invariant patch centered around its keypoint
keypoint is at position [patch_radius, patch_radius] 
"""
def nBRIEF(patch, patch_radius, vector_size, samples): 
    vector = np.zeros(vector_size, dtype=int) 
    for i in range(vector_size):        
        vector[i] = patch[patch_radius + samples[i][0], patch_radius + samples[i][1]] < patch[patch_radius + samples[i][2], patch_radius + samples[i][3]]
    return vector

"""
Steered brief, using orientation
"""
def nSBRIEF(patch, patch_radius, vector_size, samples, orientation): 
    vector = np.zeros(vector_size, dtype=bool)
    # digitize the orientation
    angles = np.array([i*(2*math.pi/30) for i in range(int(360/12))])
    theta = np.digitize(orientation, angles)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    for i in range(vector_size):
        p1r = samples[i][0]
        p1c = samples[i][1]
        p2r = samples[i][2]
        p2c = samples[i][3]
        
        sp1r = int(round(sin_theta*p1r + cos_theta*p1c)) - 1
        sp1c = int(round(cos_theta*p1r - sin_theta*p1c)) - 1
        sp2r = int(round(sin_theta*p2r + cos_theta*p2c)) - 1 
        sp2c = int(round(cos_theta*p2r - sin_theta*p2c)) - 1

        val1 = patch[patch_radius + sp1r, patch_radius + sp1c]
        val2 = patch[patch_radius + sp2r, patch_radius + sp2c]
        if val1 < val2:
            vector[i] = True
    return vector
        
