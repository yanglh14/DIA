import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

def callback_pointcloud(data):
    # Read points from the point cloud data
    points = pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True)
    for point in points:
        print(point)  # You can process the point data here

    # get all points
    points = list(pc2.read_points(data, skip_nans=True))
    points = np.array(points)

# Callback function for the depth image
def depth_callback(data):
    try:
        # Convert the ROS image to OpenCV format using a cv_bridge helper function
        bridge = CvBridge()
        depth_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

        # Output some information about the depth image
        rospy.loginfo(depth_image.shape)
        rospy.loginfo("Received depth image with max value: %f, min value: %f" % (depth_image.max(), depth_image.min()))

        # Display the depth image using OpenCV
        cv2.imshow("Depth Image", depth_image)
        cv2.waitKey(1)

    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))


def listener():
    rospy.init_node('realsense_listener', anonymous=True)
    rospy.Subscriber("/camera/depth/color/points", PointCloud2, callback_pointcloud)
    # Define the subscriber to the aligned depth image topic
    # rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_callback)

    # Prevent the script from exiting until the node is shutdown
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down depth image reader node.")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    listener()