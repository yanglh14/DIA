import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CompressedImage as msg_CompressedImage
from sensor_msgs.msg import PointCloud2 as msg_PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Imu as msg_Imu
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import struct
import ctypes
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def pc2_to_xyzrgb(point):
	# Thanks to Panos for his code used in this function.
    x, y, z = point[:3]
    rgb = point[3]

    # cast float32 to int so that bitwise operations are possible
    s = struct.pack('>f', rgb)
    i = struct.unpack('>l', s)[0]
    # you can get back the float value by the inverse operations
    pack = ctypes.c_uint32(i).value
    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)
    return x, y, z, r, g, b


class RSListener:
    def __init__(self):

        rospy.init_node('rs_listener', anonymous=True)

        self.data = []

    def visulaize(self, point_cloud):

        # Assuming you have a list/array of points named 'point_cloud'
        # with each point being a list/array [x, y, z, r, g, b]
        # For example: point_cloud = [[x1, y1, z1, r1, g1, b1], [x2, y2, z2, r2, g2, b2], ...]

        # First, convert your point cloud to a numpy array for easier manipulation
        point_cloud_np = np.array(point_cloud)

        # Split your NumPy array into positions (x, y, z) and colors (r, g, b)
        positions = point_cloud_np[:, :3]
        colors = point_cloud_np[:, 3:] / 255.0  # Assuming color channels are 0-255, normalize to 0-1 for matplotlib

        # Create a new matplotlib figure and axis.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot using the x, y, and z coordinates and the color information
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=1)  # s is the size of the points

        # Set labels for axes
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # Show the plot
        plt.savefig('test.png')

    def _pointscloudCallback(self, data):
        # Read points from the point cloud data

        self.data = np.array([pc2_to_xyzrgb(pp) for pp in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z", "rgb")) if pp[0] > 0])
        print(self.data.shape)

    def listener(self):
        rospy.Subscriber("/camera/depth/color/points", PointCloud2, self._pointscloudCallback)

        # Prevent the script from exiting until the node is shutdown
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down depth image reader node.")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rs_listener = RSListener()
    rs_listener.listener()