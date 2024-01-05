import sys
import threading

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchControllerRequest, SwitchController
from controller_manager_msgs.srv import LoadControllerRequest, LoadController
from controller_manager_msgs.srv import ListControllers, ListControllersRequest
import geometry_msgs.msg as geometry_msgs
from cartesian_control_msgs.msg import (
    FollowCartesianTrajectoryAction,
    FollowCartesianTrajectoryGoal,
    CartesianTrajectoryPoint,
)

from utils.euler import quat2euler, euler2quat

import tf
import tf2_ros
import numpy as np

# All of those controllers can be used to execute joint-based trajectories.
# The scaled versions should be preferred over the non-scaled versions.
JOINT_TRAJECTORY_CONTROLLERS = [
    "scaled_pos_joint_traj_controller",
    "scaled_vel_joint_traj_controller",
    "pos_joint_traj_controller",
    "vel_joint_traj_controller",
    "forward_joint_traj_controller",
]

# All of those controllers can be used to execute Cartesian trajectories.
# The scaled versions should be preferred over the non-scaled versions.
CARTESIAN_TRAJECTORY_CONTROLLERS = [
    "pose_based_cartesian_traj_controller",
    "joint_based_cartesian_traj_controller",
    "forward_cartesian_traj_controller",
]

# We'll have to make sure that none of these controllers are running, as they will
# be conflicting with the joint trajectory controllers
CONFLICTING_CONTROLLERS = ["joint_group_vel_controller", "twist_controller"]


class PoseClient:
    def __init__(self) -> None:
        self.tf_listener = tf.TransformListener()
        self.pub = rospy.Publisher('tool_pose', geometry_msgs.Pose, queue_size=10)
        self.pose_log = []

    def get_tool_tf(self):
        target_frame = 'tool0_controller'
        source_frame = 'base'
        try:
            self.tf_listener.waitForTransform(source_frame, target_frame, rospy.Time(0), rospy.Duration(0))
            tr_target2source = self.tf_listener.lookupTransform(source_frame, target_frame, rospy.Time(0))
            # tf_target2source = self.tf_listener.fromTranslationRotation(*tr_target2source)
            return tr_target2source 
        except Exception as e:
            print(e)
    
    def publish_tool_pose(self):
        tr_tool = self.get_tool_tf()
        if tr_tool == None:
            return
        pose = geometry_msgs.Pose(
                geometry_msgs.Vector3(tr_tool[0][0], tr_tool[0][1], tr_tool[0][2]), geometry_msgs.Quaternion(tr_tool[1][0], tr_tool[1][1], tr_tool[1][2], tr_tool[1][3])
            )
        self.pub.publish(pose)

        # logging
        time_now = rospy.Time.now()
        self.pose_log.append(np.array([time_now.to_sec(), tr_tool[0][0], tr_tool[0][1], tr_tool[0][2]]))

    def run(self):
        rate = rospy.Rate(100) # 100hz
        try:
            while not rospy.is_shutdown():
                self.publish_tool_pose()
                rate.sleep()

        except rospy.ROSInterruptException:
            rospy.loginfo('Exit Pose Client')


class TrajectoryClient:
    """Small trajectory client to test a joint trajectory"""

    def __init__(self):
        timeout = rospy.Duration(5)
        self.switch_srv = rospy.ServiceProxy(
            "controller_manager/switch_controller", SwitchController
        )
        self.load_srv = rospy.ServiceProxy("controller_manager/load_controller", LoadController)
        self.list_srv = rospy.ServiceProxy("controller_manager/list_controllers", ListControllers)
        try:
            self.switch_srv.wait_for_service(timeout.to_sec())
        except rospy.exceptions.ROSException as err:
            rospy.logerr("Could not reach controller switch service. Msg: {}".format(err))
            sys.exit(-1)

        self.joint_trajectory_controller = JOINT_TRAJECTORY_CONTROLLERS[0]
        self.cartesian_trajectory_controller = CARTESIAN_TRAJECTORY_CONTROLLERS[1]

        self.trajectory_log = []
        self.pose_init = np.array([-0.1, -0.5, 0.55, 1, 0, 0, 0])
        self.pose_init[3:] = euler2quat(np.pi, 0, 90/180*np.pi)

        self.dt = 0.01
        self.time_step = 150
        self.swing_acc_max = 2.0
        self.pull_acc_max = 0.5

    def move_to_init_pose(self):

        self.switch_controller(self.cartesian_trajectory_controller)

        # make sure the correct controller is loaded and activated
        self.goal = FollowCartesianTrajectoryGoal()
        trajectory_client = actionlib.SimpleActionClient(
            "{}/follow_cartesian_trajectory".format(self.cartesian_trajectory_controller),
            FollowCartesianTrajectoryAction,
        )

        # Wait for action server to be ready
        timeout = rospy.Duration(5)
        if not trajectory_client.wait_for_server(timeout):
            rospy.logerr("Could not reach controller action server.")
            sys.exit(-1)

        # Create initial pose
        point = CartesianTrajectoryPoint()
        point.pose = geometry_msgs.Pose(
            geometry_msgs.Vector3(self.pose_init[0], self.pose_init[1], self.pose_init[2]), geometry_msgs.Quaternion(self.pose_init[3], self.pose_init[4], self.pose_init[5], self.pose_init[6])
        )
        point.time_from_start = rospy.Duration(5.0)
        self.goal.trajectory.points.append(point)


        trajectory_client.send_goal(self.goal)
        trajectory_client.wait_for_result()

        result = trajectory_client.get_result()

        rospy.loginfo("Initialization execution finished in state {}".format(result.error_code))

    def send_cartesian_trajectory(self):
        """Creates a Cartesian trajectory and sends it using the selected action server"""
        self.switch_controller(self.cartesian_trajectory_controller)

        # make sure the correct controller is loaded and activated
        self.goal = FollowCartesianTrajectoryGoal()
        trajectory_client = actionlib.SimpleActionClient(
            "{}/follow_cartesian_trajectory".format(self.cartesian_trajectory_controller),
            FollowCartesianTrajectoryAction,
        )

        # Wait for action server to be ready
        timeout = rospy.Duration(5)
        if not trajectory_client.wait_for_server(timeout):
            rospy.logerr("Could not reach controller action server.")
            sys.exit(-1)

        trajectory = self._collect_trajectory(self.pose_init)

        time_from_start = 0

        for i, pose in enumerate(trajectory):
            time_from_start += self.dt

            # Create the Pose message
            pose_msg = geometry_msgs.Pose(
                geometry_msgs.Vector3(pose[0], pose[1], pose[2]),
                geometry_msgs.Quaternion(pose[3], pose[4], pose[5], pose[6])
            )

            # Create the CartesianTrajectoryPoint
            point = CartesianTrajectoryPoint()
            point.pose = pose_msg
            point.time_from_start = rospy.Duration(time_from_start)

            # Add to the goal
            self.goal.trajectory.points.append(point)

            # Log the pose and duration
            self.trajectory_log.append(np.array([time_from_start, pose[0], pose[1], pose[2]]))

        self.ask_confirmation(trajectory)
        rospy.loginfo(
            "Executing trajectory using the {}".format(self.cartesian_trajectory_controller)
        )
        trajectory_client.send_goal(self.goal)
        trajectory_client.wait_for_result()

        result = trajectory_client.get_result()

        rospy.loginfo("Trajectory execution finished in state {}".format(result.error_code))

        rospy.signal_shutdown("Task Done")

    def _collect_trajectory(self, cur_pose):
        """ Policy for collecting data - random sampling"""
        """ Version 2 - fixed max acceleration and float time step"""

        self.target_theta = np.random.uniform(-np.pi / 12, np.pi / 12)
        self.target_theta = np.pi / 12

        bias = np.random.uniform(0.05, 0.10)
        target_pose = np.zeros(7)
        target_pose[0] = cur_pose[0] - bias * np.cos(self.target_theta)
        target_pose[1] = cur_pose[1] - bias * np.sin(self.target_theta)
        target_pose[2] = 0.06
        target_pose[3:] = euler2quat(np.pi, 0, self.target_theta + np.pi / 2)

        cur_rot = quat2euler(cur_pose[3:])
        target_rot = quat2euler(target_pose[3:])[2] - cur_rot[2]

        # middle state sampling

        xy_trans = np.random.uniform(0.2, 0.4)
        z_ratio = np.random.uniform(0.2, 0.5)

        middle_pose = target_pose.copy()
        middle_pose[0] = middle_pose[0] - xy_trans * np.cos(target_rot)
        middle_pose[1] = middle_pose[1] - xy_trans * np.sin(target_rot)
        middle_pose[2] = cur_pose[2] + z_ratio * (target_pose[2] - cur_pose[2])

        trajectory_s2m = self._generate_trajectory(cur_pose, middle_pose, self.swing_acc_max, self.dt)
        trajectory_m2e = self._generate_trajectory(middle_pose, target_pose, self.pull_acc_max, self.dt)

        trajectory = np.concatenate((trajectory_s2m, trajectory_m2e[1:]), axis=0)

        return trajectory

    def _generate_trajectory(self, current_pos, target_pos, acc_max, dt):
        translation = (target_pos[:3] - current_pos[:3])
        rotation = np.array(quat2euler(target_pos[3:])) - np.array(quat2euler(current_pos[3:]))

        # Calculate the number of time steps
        _time_steps = max(np.sqrt(4 * np.abs(translation) / acc_max) / dt)
        time_steps = np.ceil(_time_steps).max().astype(int)

        rot_steps = rotation / time_steps

        accel_steps = int(time_steps / 2)
        decel_steps = time_steps - accel_steps

        v_max = translation * 2 / (time_steps * dt)
        accelerate = v_max / (accel_steps * dt)
        decelerate = -v_max / (decel_steps * dt)

        incremental_translation = [0, 0, 0]
        positions_xyzq = [current_pos]
        for i in range(time_steps):
            if i < accel_steps:
                # Acceleration phase
                incremental_translation = (np.divide(incremental_translation,
                                                     dt) + accelerate * dt) * dt
            else:
                # Deceleration phase
                incremental_translation = (np.divide(incremental_translation,
                                                     dt) + decelerate * dt) * dt

            # translate vertices
            _pos = positions_xyzq[-1].copy()
            _pos[:3] += incremental_translation
            _pos[3:] = euler2quat(*(rot_steps * i + quat2euler(current_pos[3:])))

            positions_xyzq.append(_pos)

        return np.array(positions_xyzq)

    ###############################################################################################
    #                                                                                             #
    # Methods defined below are for the sake of safety / flexibility of this demo script only.    #
    # If you just want to copy the relevant parts to make your own motion script you don't have   #
    # to use / copy all the functions below.                                                       #
    #                                                                                             #
    ###############################################################################################

    def ask_confirmation(self, waypoint_list):
        """Ask the user for confirmation. This function is obviously not necessary, but makes sense
        in a testing script when you know nothing about the user's setup."""
        rospy.logwarn("The robot will move to the following waypoints: \n{}".format(waypoint_list))
        confirmed = False
        valid = False
        while not valid:
            input_str = input(
                "Please confirm that the robot path is clear of obstacles.\n"
                "Keep the EM-Stop available at all times. You are executing\n"
                "the motion at your own risk. Please type 'y' to proceed or 'n' to abort: "
            )
            valid = input_str in ["y", "n"]
            if not valid:
                rospy.loginfo("Please confirm by entering 'y' or abort by entering 'n'")
            else:
                confirmed = input_str == "y"
        if not confirmed:
            rospy.loginfo("Exiting as requested by user.")
            sys.exit(0)

    def switch_controller(self, target_controller):
        """Activates the desired controller and stops all others from the predefined list above"""
        other_controllers = (
            JOINT_TRAJECTORY_CONTROLLERS
            + CARTESIAN_TRAJECTORY_CONTROLLERS
            + CONFLICTING_CONTROLLERS
        )

        other_controllers.remove(target_controller)

        srv = ListControllersRequest()
        response = self.list_srv(srv)
        for controller in response.controller:
            if controller.name == target_controller and controller.state == "running":
                return

        srv = LoadControllerRequest()
        srv.name = target_controller
        self.load_srv(srv)

        srv = SwitchControllerRequest()
        srv.stop_controllers = other_controllers
        srv.start_controllers = [target_controller]
        srv.strictness = SwitchControllerRequest.BEST_EFFORT
        self.switch_srv(srv)


if __name__ == "__main__":

    rospy.init_node("main")

    # init controller client and move to init pose
    client = TrajectoryClient()
    client.move_to_init_pose()

    # init pose client and recorde the pose of the robot during the trajectory
    pose_cli = PoseClient()
    t = threading.Thread(target=pose_cli.run)
    t.daemon = True
    t.start()

    # send trajectory and execute it
    client.send_cartesian_trajectory()

    # save log to file
    try:
        while not rospy.is_shutdown():
            rospy.sleep(1)
            print("Waiting for rospy shutdown")

        # save log to file
        np.save('../log/traj_log', pose_cli.pose_log)
        np.save('../log/traj_desired', client.trajectory_log)

        print("Saving log to file")

    except KeyboardInterrupt:
        print("Exit")