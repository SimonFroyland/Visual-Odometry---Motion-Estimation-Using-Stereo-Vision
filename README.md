From my papers abstract:

This project entails the development and evaluation of a SIFT feature-based 2D-2D visual odometry (VO) pipeline. 
The implemented VO system is evaluated using the KITTI benchmark dataset, a widely recognized resource for testing visual odometry and SLAM systems.
The evaluation includes a comparison against ground truth pose information, using selected sequences and performance metrics to assess the accuracy of the system.
In addition to the core implementation and evaluation, I conduct an extended experiment to further analyze the system's performance.
This experiment explores a comparative analysis with alternative feature detection methods which are applied to the pipeline.
The findings from this experiment provide deeper insights into the strengths, weaknesses, and potential areas for improvement of the implemented VO pipeline.

HOW TO USE:
1. Download the code 'prjoect.py' and 'functions.py'
   
2. Download the KITTY datasets:
   https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0035/2011_09_26_drive_0035_sync.zip
   https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0035/2011_09_26_drive_0018_sync.zip
   https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0035/2011_09_26_drive_0052_sync.zip
   https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip
   NB: When unziping the file structure is not in the correct order. See to it that the path is set correctly when working with the datasets.
   
3.  In 'project.py' change the path:
   basedir = 'C:/Users/Simon/Skrivebord/Vision' #Change this to your own project folder.
   date = '2011_09_26'
   drive = '0035'

4. Run 'project.py'
