# notFAST algorithm: mimicks the behavior of FAST algorithm, home-made by Gaëtan MARGUERITTE.
# Does NOT use pyramidal structure to be scale invariant
# Harris measure is NOT used to sort the N top points, instead course-mimicking nANMS (notANSM) algorithms has been made 
# Used in the nORB (notORB) algorithm
import numpy as np
from nANMS import rate_feature_point, nANMS
from matplotlib.pyplot import Circle

# returns an array of 8 pixel locations, one for each octant
def circle(i, j, x, y):
    values = np.zeros((8,2), dtype=int)
    values[0] = [i+x, j+y]
    values[1] = [i-x, j+y]
    values[2] = [i+x, j-y]
    values[3] = [i-x, j-y]

    values[4] = [i+y, j+x]
    values[5] = [i-y, j+x]
    values[6] = [i+y, j-x]
    values[7] = [i-y, j-x]
    return values

# returns an array of 16 pixels locations, depicting a 16 points bresenham circle around a point at coordinates (i,j).
def circleBres(i, j, r, nb_neighbours):
    neighbours = np.zeros((nb_neighbours,2),dtype=int)
    x = 0
    y = r
    d = 3 - (2 * r)
    while(y > x):
        x = x + 1
        if(d > 0):
            y = y - 1
            d = d + (4 * (x - y)) + 10
        else:
            d = d + (4 * x) + 6
        neighbours[(x-1)*8:x*8, :] = circle(i, j, x, y)
    return neighbours


def nFAST(img, threshold=100):
    rows, cols = img.shape
    feature_points=np.zeros(shape=(rows,cols))
    # radius of a feature point (in pixels)
    # nFAST algorithm is based on a radius of 3
    feature_radius = 3
    nb_neighbours = 16
    # number of required neighbours to have a significant variation in light intensity
    req_neighbours = 12
    # number of feature points researched
    fp_number = 50
    # threshold in light intensity
    t = threshold
    for i in range(feature_radius, rows-feature_radius):
        for j in range (feature_radius, cols-feature_radius):
            neighbours = circleBres(i, j, feature_radius, nb_neighbours)
            pixl = int(img[i, j])
            good_neighbours = 0
            # FAST checks the 4 first values
            for k in range(4):
                pos = neighbours[k]
                neighb = int(img[pos[0], pos[1]])
                if(neighb > pixl + t):
                    good_neighbours = good_neighbours + 1
                elif(neighb < pixl - t):
                    good_neighbours = good_neighbours - 1
            for k in range(4, nb_neighbours):
                pos = neighbours[k]
                neighb = int(img[pos[0], pos[1]])
                if(good_neighbours >= 3):
                    if(neighb < pixl - t):
                        break 
                    elif(neighb > pixl + t):
                        good_neighbours = good_neighbours + 1
                elif(good_neighbours <= -3):
                    if(neighb > pixl + t):
                        break
                    elif(neighb < pixl - t):
                        good_neighbours = good_neighbours - 1
                else:
                    break
            if(good_neighbours >= req_neighbours or good_neighbours <= req_neighbours*-1):
                feature_points[i, j] = rate_feature_point(pixl, neighb, nb_neighbours)

    return feature_points
                
def draw_feature_points(fps, feature_radius, shape, ax):
    for i in range(shape[0]):
        for j in range (shape[1]):
            if(fps[i, j] != 0):
                circ = Circle((j, i), feature_radius, fill=False, color='g')
                ax.add_patch(circ)


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    import time
    from matplotlib.patches import Circle

    # execute only if run as a script
    # load a grayscaled image
    img = cv2.imread('lizard.jpg', 0)

    t = 100
    feature_radius = 3
    start = time.time()
    feature_points = nFAST(img, t)
    end = time.time()

    print("worktime to find interest points through nFAST = ", end-start)
    print("image shape (rows, cols) = ", img.shape)
    print("feature radius =", feature_radius)
    print("feature threshold =", t)
    print("number of feature points detected =", len(np.nonzero(feature_points)[0]))

    fig, (ax, ax2) = plt.subplots(2) 
    # apply plt color scheme to image
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # display image
    ax.imshow(RGB_img)
    
    draw_feature_points(feature_points, feature_radius, img.shape, ax)

    # number of maximum feature points searched
    max_fps = 300
    start = time.time()
    # starting suppression radius of ANSM set at 10 times the feature point size
    new_feature_points = nANMS(feature_points, shape=img.shape, nb=max_fps, radius=feature_radius*10)
    end = time.time()

    print("worktime to use nANMS = ", end-start)
    print("number of feature points detected = ", len(np.nonzero(new_feature_points)[0]))

    ax2.imshow(RGB_img)
    draw_feature_points(new_feature_points, feature_radius, img.shape, ax2)

    plt.show()



