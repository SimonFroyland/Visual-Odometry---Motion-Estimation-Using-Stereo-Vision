#====================================================
# Functions
#====================================================
import cv2
import numpy as np
import pykitti
import cv2
from matplotlib import pyplot as plt
from datetime import datetime

from spatialmath import *
from spatialmath.base import *
from spatialmath.base import sym
from spatialgeometry import *

#---------------------------------------------
# Feature detection
#---------------------------------------------
def detect_features(img, method='sift'):
    """Detect features in a grayscale image."""
    
    if method == 'sift':
        # detect features
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
    
    elif method == 'orb':
        # detect features
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(img, None)
    
    elif method == 'brisk':
        # detect features
        brisk = cv2.BRISK_create()
        kp, des = brisk.detectAndCompute(img, None)

    elif method == 'akaze':
        # detect features
        akaze = cv2.AKAZE_create()
        kp, des = akaze.detectAndCompute(img, None)
    
    elif method == 'fast':
        # FAST is a corner detection method, not directly providing descriptors
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(img, None)
        # No descriptors for FAST, returning None
        des = None

    else:
        # not implemented error
        raise NotImplementedError('Unknown feature detection method.')
        
    return kp, des


#---------------------------------------------
# Feature matching
#---------------------------------------------
def match_features(des1, des2):
    """Match features between two images."""
    
    # match features
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    
    return matches

#---------------------------------------------
# Feature detection and matching
#---------------------------------------------
def detect_and_match(img1, img2, m='sift'):
    """Detect features in two grayscale images and match features."""
    kp1, des1 = detect_features(img1, method=m)
    kp2, des2 = detect_features(img2, method=m)
    match_features(des1, des2)

#---------------------------------------------
# Disparity map
#---------------------------------------------
def disparity(left, right):
    """Creates disparity map between right and left  grayscale image"""
    left = np.array(left)
    right = np.array(right)

    # create the Semi-Global Block Matching object
    stereo = cv2.StereoSGBM_create(blockSize=9, numDisparities=85, P2=32*3*9*9, P1=8*3*9*9, disp12MaxDiff=100, uniquenessRatio=None, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    # compute the disparity map
    disparity = stereo.compute(left, right)

    # Scale the outputs to actual pixel values
    disparity = disparity.astype(np.float32) / 16.0
    disparity = cv2.medianBlur(disparity, 5)

    return disparity

#---------------------------------------------
# 2D2D motion.
#---------------------------------------------
def motion_from_2D2D(img1, img2, K):
    """Estimate the motion between two images using the 2D-2D method."""
    
    # detect features in both images
    kp1, des1 = detect_features(img1, 'sift')
    kp2, des2 = detect_features(img2, 'sift')

    # match features between the two images
    matches = match_features(des1, des2)

    # estimate the essential matrix
    E, mask = cv2.findEssentialMat(
        np.array([kp1[m.queryIdx].pt for m in matches]),
        np.array([kp2[m.trainIdx].pt for m in matches]),
        cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # recover the pose from the essential matrix
    points, R, t, mask = cv2.recoverPose(E,
        np.array([kp1[m.queryIdx].pt for m in matches]),
        np.array([kp2[m.trainIdx].pt for m in matches]))
    
    # we return a SE3 object describing the pose of the second camera in the frame of the first camera
    # we also return the mask of inliers
    return SE3(t) * SE3(SO3(R)), mask

#---------------------------------------------
# Disparity map
#---------------------------------------------
def estimate_depth(disparity_map, f, B):
    """Estimate depth from disparity map."""
    # Avoid division by zero
    disparity_map[disparity_map == 0] = 0.01
    depth = (f * B) / disparity_map

    #---------------------------------------------
    # depth thresholding
    #---------------------------------------------
    depth[depth >  50] = -1 # for highway: depth[depth <  35] = -1
    depth[depth < 0 ] = -1
    
    return depth

#---------------------------------------------
# Time difference
#---------------------------------------------
def time_difference(timestamps):
    datetime_objects = []

    for ts in timestamps:
        if isinstance(ts, datetime):
            datetime_objects.append(ts)
        else:
            datetime_objects.append(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f"))

    # Calculate the time differences in seconds
    time_diffs_in_seconds = [(datetime_objects[i] - datetime_objects[i-1]).total_seconds() 
                             for i in range(1, len(datetime_objects))]

    return time_diffs_in_seconds