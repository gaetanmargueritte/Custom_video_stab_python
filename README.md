# Custom_video_stab_python
Author: Gaëtan MARGUERITTE.
A homemade video motion smoothing, using Python. Made during a project of Computer Vision, during a course followed at Keio University.

## Usage of the program
`python3 main.py` will launch the full treatment of a video file `video_input.mp4`, resulting in an output `video_out.mp4`. 

## Main ideas
- Mimicks the behaviour of ORB feature matcher, without scaling invariance. Created using solely the original paper "ORB: an efficient alternative to SIFT or SURF" by Ethan Rublee, Vincent Rabaud, Kurt Konolige, Gary Bradski.
- Using the Lucas-Kanade (sparse) optical flow, compute each feature point motion
- Deduces each frame motion from the feature point motions with a naive algorithm
- Smoothes the motion to stabilize the result

## Technologies used: 
* Python3
* numpy (array management and array operations)
* scipy (signal functions)
* math (mathematical functions)
* Opencv (video input/output manipulations)


## Side note
This algorithm is a personal project without any commercial use, and is slow, due to its naive and straight-forward nature.
