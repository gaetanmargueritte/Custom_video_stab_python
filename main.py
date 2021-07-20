import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

from FeaturePoint import FeaturePoint
from nORB import nORB
from optical_flow import nLK, get_naive_translation, smooth_motion
from warp import warp_affine 

print(cv2.__version__)

if __name__ == "__main__":
    # execute only if run as a script
    # read input video
    cap = cv2.VideoCapture('video_input.mp4')

    # get frames count
    #n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames = 500
    
    print("Number of frames of the video =", n_frames)
    # Get width and height of video stream
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    # Set up output video
    out = cv2.VideoWriter('video_out.mp4', fourcc, 30, (w, h))
    # Read first frame
    _, prev = cap.read() 
    
    # Convert frame to grayscale
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    fps = None
    translations = np.zeros((n_frames-1, 2), np.float32)
    start=time.time()
    # slow to compute : make only 120 frames
    for i in range(n_frames-1):
        print("==== Frame ", i, " / ", n_frames, " ====")
        # Read next frame
        success, curr = cap.read()
        if not success:
            break

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

        fps, fps2, matches = nORB(prev_gray, curr_gray, True, fps)

        used_fps = [fps[matches[i][0]] for i in range(len(matches))]
        used_fps2 = [fps2[matches[i][1]] for i in range(len(matches))]
        u, v = nLK(prev_gray, curr_gray, fps, fps2, matches)

        translation = get_naive_translation(used_fps, u, v)
        translations[i] = translation
        print(translations[i])

        # Move to next frame
        prev_gray = curr_gray

    # see trajectory of image over time through cumulative sum
    trajectory = np.cumsum(translations, axis=0)
    smoothed_trajectory = np.zeros(shape=trajectory.shape)
    smoothed_trajectory[:, 0] = smooth_motion(trajectory[:, 0], 3)
    smoothed_trajectory[:, 1] = smooth_motion(trajectory[:, 1], 3)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    trajectory = translations + difference
    
    end = time.time()
    print("Video motion estimation working time = ", end-start)
    # write in output stream
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # Write n_frames-1 transformed frames
    for i in range(n_frames-2):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        # Extract transformations from the new transformation array
        motion_x = trajectory[i,0]
        motion_y = trajectory[i,1]
        
        translation_matrix = np.zeros((2,3), np.float32)
        translation_matrix[0,0] = translation_matrix[1,1] = 1
        translation_matrix[0,2] = motion_x
        translation_matrix[1,2] = motion_y
        
        # Apply affine wrapping to the given frame
        frame_stabilized = warp_affine(frame, translation_matrix, True) 

        # frame_stabilized = cv2.cvtColor(frame_stabilized, cv2.COLOR_BGR2GRAY)
        
        # Write the frame to the file
        frame_out = cv2.hconcat([frame, frame_stabilized])
        
        cv2.imshow("Before and After", frame_out)
        cv2.waitKey(10)
        out.write(frame_out)

    
    plt.show()

