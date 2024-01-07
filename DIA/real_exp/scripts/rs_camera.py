from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import struct
import ctypes
import message_filters
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.rs_utils import object_detection

class RSListener:
    def __init__(self):

        rospy.init_node('rs_listener', anonymous=True)
        self.bridge = CvBridge()

        self.mask = None
        self.points = None

    def _image_callback(self, data):
        try:
            # Convert the image to OpenCV format
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        cv2.imwrite('../log/rgb.png', image)
        self.mask = object_detection(image)

    def _depth_callback(self, depth_image_msg, camera_info_msg):

        try:
            if self.mask is None:
                return
            # Convert the ROS image to OpenCV format using a cv_bridge helper function
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, depth_image_msg.encoding)

            # Get the camera intrinsic parameters
            fx = camera_info_msg.K[0]
            fy = camera_info_msg.K[4]
            cx = camera_info_msg.K[2]
            cy = camera_info_msg.K[5]

            # Generate the point cloud
            height, width = depth_image.shape

            # Create a meshgrid of pixel coordinates
            u, v = np.meshgrid(np.arange(width), np.arange(height))

            # Apply the mask to the depth image
            # Only consider pixels where the mask is 255
            Z = np.where(self.mask == 255, depth_image, 0)

            # Flatten the arrays for vectorized computation
            u, v, Z = u.flatten(), v.flatten(), Z.flatten() * 0.001  # Depth scale (mm to meters)

            # Filter out the points with zero depth after masking
            valid_indices = Z > 0
            u, v, Z = u[valid_indices], v[valid_indices], Z[valid_indices]

            # Compute the X, Y world coordinates
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy

            # Stack the coordinates into a point cloud
            point_cloud = np.vstack((X, Y, Z)).transpose()
            self.points = point_cloud

            # convert camera coordinate to robot coordinate


        except CvBridgeError as e:
            print(e)

    def run(self):
        # Create a subscriber to the aligned depth image topic
        # self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self._depth_callback)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self._image_callback)

        # Use message_filters to subscribe to the image and camera info topics
        depth_image_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        camera_info_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo)

        ts = message_filters.TimeSynchronizer([depth_image_sub, camera_info_sub], 10)
        ts.registerCallback(self._depth_callback)

        # Prevent the script from exiting until the node is shutdown
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down depth image reader node.")

if __name__ == '__main__':
    rs_listener = RSListener()
    rs_listener.run()