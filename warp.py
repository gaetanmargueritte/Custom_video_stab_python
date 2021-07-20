# naive affine warper, home-made by Gaëtan MARGUERITTE
import time
import numpy as np

def warp_affine(img, matrix, verbose=False):
    start = time.time()
    output = np.zeros(shape=img.shape, dtype=img.dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p = np.dot(matrix, [i, j, 1])
            i_prime = int(p[0])
            j_prime = int(p[1])
            if 0 <= i_prime and i_prime < img.shape[0] and 0 <= j_prime and j_prime < img.shape[1]:
                output[i_prime, j_prime] = img[i, j]
    end = time.time()
    
    if verbose:
        print()
        print("Worktime to compute optical flows = ", end-start)
        print("u vector is ", matrix[0, 2])
        print("v vector is ", matrix[1, 2])

    return output
