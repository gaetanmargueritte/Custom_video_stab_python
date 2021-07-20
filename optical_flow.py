# naive optical flow utilities, home-made by Gaëtan MARGUERITTE

import numpy as np
import time


from FeaturePoint import FeaturePoint
from scipy import signal

"""
not Lucas Kanade, estimates the optical flow of 3x3 patches around a keypoint
"""
def nLK(img, img2, fps, fps2, matches, verbose=False):
    start = time.time()

    # convolution of full images
    Ix = signal.convolve2d(img,[[-0.25,0.25],[-0.25,0.25]],'same') + signal.convolve2d(img2,[[-0.25,0.25],[-0.25,0.25]],'same') 
    Iy = signal.convolve2d(img,[[-0.25,-0.25],[0.25,0.25]],'same') + signal.convolve2d(img2,[[-0.25,-0.25],[0.25,0.25]],'same')
    It = signal.convolve2d(img,[[0.25,0.25],[0.25,0.25]],'same') + signal.convolve2d(img2,[[-0.25,-0.25],[-0.25,-0.25]],'same')
    
    # returned vector for each point
    u = np.zeros(shape=img.shape)
    v = np.zeros(shape=img.shape)
    for i in range(len(matches)):
        i, j = fps[matches[i][0]].position
        
        # 3*3 window, 9 elements per dimension
        IX = ([Ix[i-1,j-1],Ix[i,j-1],Ix[i-1,j-1],Ix[i-1,j],Ix[i,j],Ix[i+1,j],Ix[i-1,j+1],Ix[i,j+1],Ix[i+1,j-1]]) 
        IY = ([Iy[i-1,j-1],Iy[i,j-1],Iy[i-1,j-1],Iy[i-1,j],Iy[i,j],Iy[i+1,j],Iy[i-1,j+1],Iy[i,j+1],Iy[i+1,j-1]]) 
        IT = ([It[i-1,j-1],It[i,j-1],It[i-1,j-1],It[i-1,j],It[i,j],It[i+1,j],It[i-1,j+1],It[i,j+1],It[i+1,j-1]]) 
		
        #  minimum least squares solution
        ssd = (IX, IY)
        ssd = np.matrix(ssd)
        ssd_t = np.array(np.matrix(ssd)) 
        ssd = np.array(np.matrix.transpose(ssd)) 
	
        a1 = np.dot(ssd_t,ssd) 
        a2 = np.linalg.pinv(a1)
        a3 = np.dot(a2,ssd_t)

        val = np.dot(a3, IT)
        u[i, j] = val[0]
        v[i, j] = val[1]
                                    
    end = time.time()
    if verbose:
        print("Worktime to compute optical flows = ", end-start)

    return u,v
    

"""
returns a naive way to find global translation of an image: mean value of vectors u and v
"""
def get_naive_translation(fps, u, v):
    positions = np.array([fps[i].position for i in range(len(fps))])
    mean_u = np.mean([u[positions[i][0], positions[i][1]] for i in range(len(fps))])
    mean_v = np.mean([v[positions[i][0], positions[i][1]] for i in range(len(fps))])
    return [mean_u, mean_v]


"""
smoothes a motion over a specified range of points
"""
def smooth_motion(motion, radius):
    patch_size = (2 * radius) + 1
    motion_filter = np.ones(patch_size)/patch_size
    padding = np.pad(motion, (radius, radius), 'edge')
    padded_result = np.convolve(padding, motion_filter, mode='same')
    result = padded_result[radius:len(padded_result)-radius]
    return result
