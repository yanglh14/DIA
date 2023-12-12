import rospy
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import numpy as np

class RealSensePointCloud:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('realsense_pointcloud_visualizer')

        # Create a subscriber to the RealSense point cloud ROS topic
        # Make sure to use the correct topic where your RealSense camera publishes the point cloud data
        self.pc_subscriber = rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.pointcloud_callback, queue_size=1)

        # Initialize Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='RealSense Point Cloud')

        # Initialize point cloud variable
        self.pcd = o3d.geometry.PointCloud()

        self.voxel_size = 0.01  # Adjust this value as needed

        # Variable to control the main loop
        self.is_running = True

    def pointcloud_callback(self, msg):
        try:
            # Convert ROS PointCloud2 message to array of xyz points
            pc_array = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = np.array(list(pc_array))

            if points.size == 0:
                rospy.logwarn("Received an empty point cloud.")
                return

            # Update point cloud
            self.pcd.clear()
            self.pcd.points = o3d.utility.Vector3dVector(points)

            # Voxel downsample the point cloud
            self.pcd = self.pcd.voxel_down_sample(voxel_size=self.voxel_size)

            # Update the visualizer
            self.update_visualizer()
        except Exception as e:
            rospy.logerr("An error occurred in pointcloud_callback: %s", str(e))

    def update_visualizer(self):
        self.vis.clear_geometries()  # Clear old geometries
        self.vis.add_geometry(self.pcd)  # Add the current point cloud
        self.vis.poll_events()  # Update the visualizer events
        self.vis.update_renderer()  # Render the point cloud

    def run(self):
        # Main loop
        while not rospy.is_shutdown() and self.is_running:
            rospy.spin()

        # If ROS is shut down or the script is stopped, close the visualizer window
        self.vis.destroy_window()

# Main function
if __name__ == "__main__":
    try:
        pointcloud_visualizer = RealSensePointCloud()
        pointcloud_visualizer.run()
    except rospy.ROSInterruptException:
        pass