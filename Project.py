import Functions as fnc
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

#-------------------------------------------------
# Read the dataset sequence
#-------------------------------------------------
basedir = 'C:/Users/Simon/Skrivebord/Vision'
date = '2011_09_26'
drive = '0035'

#-------------------------------------------------
# Load the data
#-------------------------------------------------
data = pykitti.raw(basedir, date, drive)

#-------------------------------------------------
# Loads timestamps of the pictures and convert to
# seconds past between each image.
#-------------------------------------------------
timestamps = data.timestamps
time_diffs = fnc.time_difference(timestamps)


relative_poses = []
relative_poses.append(SE3())
masks = []
K = data.calib.K_cam3
img1 = None

#-------------------------------------------------
# Iterate over all the images stored in data.
#-------------------------------------------------
for left, right in data.rgb:
    #----------------------------------------------------
    # convert the image into opencv format and grayscale
    #----------------------------------------------------
    right = cv2.cvtColor(np.array(right), cv2.COLOR_RGB2GRAY)    

    if img1 is None:
        img1 = right       
        continue
    else:
        img2 = right

        #-------------------------------------------------
        # estimate the motion between two consecutive 
        # images, returns a SE3 object
        #-------------------------------------------------
        T, mask = fnc.motion_from_2D2D(img1, img2, K)
        relative_poses.append(T)
        masks.append(mask)

        img1 = img2    

        if len(relative_poses) % 25 == 0:
            print("Processed %d images" % len(relative_poses))

#-------------------------------------------------
# First camera is looking along the x-axis of 
# the world frame and is located at (0,0,0)
#-------------------------------------------------
c1Rw = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
c1Tw = SE3(SO3(c1Rw)) 
wTc1 = c1Tw.inv()       # Pose of camera 2 in the world frame

#-------------------------------------------------
# List of world frame camera poses.
#-------------------------------------------------
trajectory = [wTc1]

#-------------------------------------------------
# Compute poses in the world frame.
#-------------------------------------------------
f = data.calib.K_cam3[0, 0]
B = 0.54
speed = []
for i, c2Tc1 in enumerate(relative_poses[1:]):     
    
    c1Tc2 = c2Tc1.inv() 
    
    #-------------------------------------------------
    # Load left and right images.
    #-------------------------------------------------
    left, right = data.get_rgb(i)
    left = cv2.cvtColor(np.array(left), cv2.COLOR_RGB2GRAY)  
    right = cv2.cvtColor(np.array(right), cv2.COLOR_RGB2GRAY)
    
    #-------------------------------------------------
    # create disparity
    #-------------------------------------------------
    disp = fnc.disparity(left, right)

    #-------------------------------------------------
    # Estimate depth in meters from
    # the disparity map
    #-------------------------------------------------
    depth = fnc.estimate_depth(disp, f, B)

    #-------------------------------------------------
    # Create keypoints and descriptors.
    #-------------------------------------------------
    # (This is also done in motion estimation.
    # This is redundant here, but i won't change the 
    # code so late in the process. It could save 
    # considerable computation time though.)
    #-------------------------------------------------
    if i == 0:
        kp1, des1 = fnc.detect_features(right, method='sift')
        depth1 = depth
    else:
        kp2, des2 = fnc.detect_features(right, method='sift')
    
        #-------------------------------------------------
        # match features between
        # consecutive right images.
        #-------------------------------------------------
        matches = fnc.match_features(des1, des2)

        #-------------------------------------------------
        # Cook some coordinates from 
        # the matches. between 
        # consecutive images
        # in the right camera.
        #-------------------------------------------------
        right1_points = []
        right2_points = []
        for match in matches:
            right1_idx = match.queryIdx
            right2_idx = match.trainIdx
            [x1, y1] = kp1[right1_idx].pt
            [x2, y2] = kp2[right2_idx].pt
            right1_points.append([x1, y1])
            right2_points.append([x2, y2])

        #-------------------------------------------------
        # Finding all depths in meters
        # at the keypoint coordinates
        #-------------------------------------------------
        depth_points1 = []
        depth_points2 = []
        for x, y in right1_points:
            depth_points1.append(depth1[int(round(y, 1)), int(round(x, 1))])
        
        for x, y in right2_points:
            depth_points2.append(depth[int(round(y, 1)), int(round(x, 1))])

        #-------------------------------------------------
        # Ensuring all depths out of
        # bound is eliminated
        #-------------------------------------------------
        depth_points1, depth_points2 = zip(*[(d1, d2) for d1, d2 in zip(depth_points1, depth_points2) if d1 != -1 and d2 != -1])
        depth_points1 = list(depth_points1)
        depth_points2 = list(depth_points2)

        #-------------------------------------------------
        # Creating the distance in
        # meter the car traveled
        # between the images.
        #-------------------------------------------------
        scale = np.median(np.subtract(depth_points1, depth_points2))
        print(scale)
        depth1 = depth
        kp1 = kp2
        des1 = des2
        speed.append(scale/time_diffs)
        #-------------------------------------------------
        # Apply scale to translation vector
        #-------------------------------------------------
        c1Tc2.t *= scale

    #-------------------------------------------------
    # if camera is moving backwards invert the 
    # relative pose
    #-------------------------------------------------
    if c1Tc2.t[2] < 0:
        c1Tc2.t[2] *= -1
    #-------------------------------------------------
    # Append pose of the current camera in the
    # world frame
    #-------------------------------------------------
    trajectory.append(trajectory[-1] * c1Tc2)

#-------------------------------------------------
# m/s
#-------------------------------------------------
#print(speed)
# ===== PLOTTING =====
#-------------------------------------------------
# Use first ground truth pose align world and 
# ground truth frames
#-------------------------------------------------
gTw = SE3(data.oxts[0].T_w_imu)

#-------------------------------------------------
# Trajectory is in world frame, transforming
# it to the ground truth frame
#-------------------------------------------------
trajectory = [gTw * T for T in trajectory]

#-------------------------------------------------
# plot the estimated trajectory
#-------------------------------------------------
traj = np.array([ [T.t[0] for T in trajectory], [T.t[1] for T in trajectory] ])
plt.plot(traj[0,:], traj[1,:],'b-', label='Estimated Trajectory')

#-------------------------------------------------
# plot the ground truth into the same plot
#-------------------------------------------------
gt = np.array([ [oxts.T_w_imu[0,3] for oxts in data.oxts], [oxts.T_w_imu[1,3] for oxts in data.oxts] ])
plt.plot(gt[0,:], gt[1,:],'r-', label='Ground Truth')
plt.axis('equal')
plt.xlabel('x'); plt.ylabel('y'); plt.title('Trajectory')
plt.grid()
plt.legend()

#-------------------------------------------------
# Plotting error between the estimated and ground
# truth trajectories, the number of matches and 
# inliners, returned in `masks`
#-------------------------------------------------
error = np.linalg.norm(traj - gt, axis=0)
plt.figure()
plt.plot(error*1000,'r', label='ATE [mm]')
plt.plot([len(m) for m in masks], label='Matched Features')
plt.plot([np.sum(m)/255 for m in masks], '--', label='Inliers')
plt.legend()
plt.grid()
plt.show()

#-------------------------------------------------
# print some error metrics
# ATE = Absolute Pose Error 
# (but we only calculate it on the XY plane here)
# RMSE = Root Mean Squared Error
#-------------------------------------------------
print(f'Mean ATE: \t{np.mean(error)*1000:.2f} mm')
print(f'Median ATE: \t{np.median(error)*1000:.2f} mm')
print(f'Max ATE: \t{np.max(error)*1000:.2f} mm')
print(f'RMSE: \t\t{np.sqrt(np.mean((traj - gt)**2))*1000:.2f} mm')
