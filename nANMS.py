# notANMS algorithm: provides utility methods mimicking ANMS algorithm behavior, home-made by Gaëtan MARGUERITTE
# used in the nFAST (notFAST) algorithm
import numpy as np

"""
the rating of a feature point is the sum of absolute differences between light intensity at feature point and 16 surrounding pixel values
"""
def rate_feature_point(fp, neighbours, nb_neighbours):
    rating = np.sum(np.abs(neighbours - fp))
    return rating

"""
linear complexity of the image shape
"""
def nNMS(fps, shape, r=10):
    result = np.zeros(shape, dtype=int)
    row = 0
    while row < shape[0]:
        row_end = min(shape[0], row+r)
        col = 0
        while col < shape[1]:
            col_end = min(shape[1], col+r)
            maxima = [0, 0]
            value = 0
            for i in range(row, row_end):
                for j in range(col, col_end):
                    pixl = fps[i, j]
                    if pixl > value * 1.1: # needs to be significantly larger (+10%)
                        value = pixl
                        maxima = [i, j]
            if value:
                result[maxima[0], maxima[1]] = value
            col = col + r
        row = row + r
    return result

"""
Adaptative nNMS, will repeat NMS until number of detected feature points is closest of nb
"""
def nANMS(fps, shape, nb=500, radius=0):
    result = fps
    r = radius 
    while len(np.nonzero(result)[0]) > nb:
        # radius = radius + 1
        # greedy radius increase used to be faster
        radius = radius + radius
        result = nNMS(result, shape, radius)
    return result
    
