from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

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

from utils.rs_utils import object_detection
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
        self.bridge = CvBridge()

        self.mask = None
        self.data = []

    def _pc_callback(self, data):
        # Read points from the point cloud data

        cloud_array = np.array(
            list(pc2.read_points(data, field_names=("x", "y", "z", "rgb"), skip_nans=False)))

        img = []
        for point in cloud_array:
            x, y, z, r, g, b = pc2_to_xyzrgb(point)
            self.data.append([x, y, z, r, g, b])
            img.append([r, g, b])
        img = np.array(img).reshape((480, 848, 3))
        # save img
        cv2.imwrite('../log/rgb.png', img)

        if self.mask is not None:
            height_mask, width_mask = self.mask.shape
            height,width = data.height, data.width
            cloud_array = cloud_array.reshape((height, width, -1))
            bias_height = int((height - height_mask)/2)
            bias_width = int((width - width_mask)/2)
            cloud_array = cloud_array[bias_height:bias_height+height_mask, bias_width:bias_width+width_mask, :3]
            np.save('../log/rs_data.npy', cloud_array)

        # # Initialize an empty list to hold points that are dark green
        # obj_points = []

        # Iterate over each point in the point cloud
        # for point in pc2.read_points(data, field_names=("x", "y", "z", "rgb"), skip_nans=True):
        #     # Extract RGB values
        #     # The point_cloud2.read_points method returns RGB as a packed float, which is why we need to convert it
        #     rgb_packed = struct.unpack('I', struct.pack('f', point[3]))[0]
        #     r = (rgb_packed >> 16) & 0x0000ff
        #     g = (rgb_packed >> 8) & 0x0000ff
        #     b = (rgb_packed) & 0x0000ff
        #     x, y, z = point[:3]
        #
        #     if z < 0.5 or z > 1.5 or y < -0.3 or y > 0.3 or x < -0.3 or x > 0.3:
        #         continue
        #
        #     if self.mask is not None:
        #         continue
        # print(len(obj_points))
        # np.save('../log/rs_data.npy', obj_points)

    def _image_callback(self, data):
        try:
            # Convert the image to OpenCV format
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.mask = object_detection(image)
        print(self.mask.shape)
    def run(self):

        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self._image_callback)
        self.pc_sub = rospy.Subscriber("/camera/depth/color/points", PointCloud2, self._pc_callback)

        # Prevent the script from exiting until the node is shutdown
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down depth image reader node.")

if __name__ == '__main__':
    rs_listener = RSListener()
    rs_listener.run()