#!/usr/bin/env python

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# Copyright 2021 FZI Forschungszentrum Informatik
# Created on behalf of Universal Robots A/S
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -- END LICENSE BLOCK ------------------------------------------------
#
# ---------------------------------------------------------------------
# !\file
#
# \author  Felix Exner mauch@fzi.de
# \date    2021-08-05
#
#
# ---------------------------------------------------------------------
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

import tf
import tf2_ros
import numpy as np

# Compatibility for python2 and python3
if sys.version_info[0] < 3:
    input = raw_input

# If your robot description is created with a tf_prefix, those would have to be adapted
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

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
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.pub = rospy.Publisher('tool_pose', geometry_msgs.Pose, queue_size=10)
        self.pose_log = []

    def get_tool_tf(self):
        time = rospy.Time(0)
        target_frame = 'tool0_controller'
        source_frame = 'base'
        try:
            self.tf_listener.waitForTransform(target_frame, source_frame, time, rospy.Duration(0))
            tr_target2source = self.tf_listener.lookupTransform(target_frame, source_frame, time)
            # tf_target2source = self.tf_listener.fromTranslationRotation(*tr_target2source)
            return tr_target2source 
        except Exception as e:
            print(e)
            tf_target2source = None
    
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
        self.cartesian_trajectory_controller = CARTESIAN_TRAJECTORY_CONTROLLERS[0]

        self.trajectory_log = []
        self.pose_init = np.array([-0.4, -0.4, 0.366, 1, 0, 0, 0])
        self.pose_end = np.array([-0.4, -0.5, 0.066, 1, 0, 0, 0])

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

        self.dt = 0.01
        self.time_step = 100

        trajectory = self.collect_trajectory(self.pose_init, self.pose_end)
        print(trajectory)

        time_from_start = 0
        self.pose_list = []
        self.duration_list = []

        for pose in trajectory:
            time_from_start = time_from_start + self.dt

            self.pose_list.append(
                geometry_msgs.Pose(geometry_msgs.Vector3(pose[0], pose[1], pose[2]),
                                   geometry_msgs.Quaternion(self.pose_init[3], self.pose_init[4], self.pose_init[5], self.pose_init[6])
            ))
            self.duration_list.append(time_from_start)

        for i, pose in enumerate(self.pose_list):
            point = CartesianTrajectoryPoint()
            point.pose = pose
            point.time_from_start = rospy.Duration(self.duration_list[i])
            self.goal.trajectory.points.append(point)

            # logging
            self.trajectory_log.append(np.array([self.duration_list[i], pose.position.x, pose.position.y, pose.position.z]))
            
        self.ask_confirmation(self.pose_list)
        rospy.loginfo(
            "Executing trajectory using the {}".format(self.cartesian_trajectory_controller)
        )
        trajectory_client.send_goal(self.goal)
        trajectory_client.wait_for_result()

        result = trajectory_client.get_result()

        rospy.loginfo("Trajectory execution finished in state {}".format(result.error_code))


    def collect_trajectory(self, pose_init, pose_end):
        """ Policy for collecting data - random sampling"""

        current_picker_position = np.array([[pose_init[1], pose_init[2], pose_init[0]-0.1],[pose_init[1], pose_init[2], pose_init[0]+0.1]])

        target_picker_position = np.array([[pose_end[1], pose_end[2], pose_end[0]-0.1],[pose_end[1], pose_end[2], pose_end[0]+0.1]])

        middle_position_step_ratio = np.random.uniform(0.3, 0.7)
        middle_position_xy_translation = np.random.uniform(0.1, 0.2)
        middle_position_z_ratio = np.random.uniform(0.2, 0.5)

        norm_direction = np.array([target_picker_position[1, 2] - target_picker_position[0, 2],
                                   target_picker_position[0, 0] - target_picker_position[1, 0]]) / \
                         np.linalg.norm(np.array([target_picker_position[1, 2] - target_picker_position[0, 2],
                                                  target_picker_position[0, 0] - target_picker_position[1, 0]]))
        middle_state = target_picker_position.copy()
        middle_state[:, [0, 2]] = target_picker_position[:, [0, 2]] + -middle_position_xy_translation * norm_direction
        middle_state[:, 1] = current_picker_position[:, 1] + middle_position_z_ratio * (
                    target_picker_position[:, 1] - current_picker_position[:, 1])

        trajectory_start_to_middle = self._trajectory_generation(current_picker_position, middle_state,
                                                              int(self.time_step * middle_position_step_ratio))

        trajectory_middle_to_target = self._trajectory_generation(middle_state, target_picker_position,
                                                            self.time_step - int(
                                                                self.time_step * middle_position_step_ratio))

        # cat trajectory_xy and trajectory_z
        trajectory = np.concatenate((trajectory_start_to_middle, trajectory_middle_to_target[1:]), axis=0)
        trajectory = trajectory.reshape(trajectory.shape[0], -1)

        trajectory = (trajectory[:,:3] + trajectory[:,3:6])/2

        trajectory[:,[0,1,2]] = trajectory[:,[2,0,1]]

        return trajectory

    def _trajectory_generation(self, current_picker_position, target_picker_position, time_steps):

        """ Policy for trajectory generation based on current and target_picker_position"""

        # select column 1 and 3 in current_picker_position and target_picker_position
        initial_vertices_xy = current_picker_position[:, [0, 2]]
        final_vertices_xy = target_picker_position[:, [0, 2]]

        # calculate angle of rotation from initial to final segment in xy plane
        angle = np.arctan2(final_vertices_xy[1, 1] - final_vertices_xy[0, 1],
                           final_vertices_xy[1, 0] - final_vertices_xy[0, 0]) - \
                np.arctan2(initial_vertices_xy[1, 1] - initial_vertices_xy[0, 1],
                           initial_vertices_xy[1, 0] - initial_vertices_xy[0, 0])

        # number of steps
        steps = time_steps

        # calculate angle of rotation for each step
        rotation_angle = angle / steps

        # translation vector: difference between final and initial centers
        translation = (target_picker_position.mean(axis=0) - current_picker_position.mean(axis=0))

        # calculate incremental translation
        incremental_translation = [0, 0, 0]

        # initialize list of vertex positions
        positions_xzy = [current_picker_position]

        # divide the steps into two parts
        accelerate_steps = steps // 2
        decelerate_steps = steps - accelerate_steps

        v_max = translation * 2 / ((accelerate_steps + decelerate_steps) * self.dt)
        acc_accelerate = v_max / (accelerate_steps * self.dt)
        acc_decelerate = -v_max / (decelerate_steps * self.dt)

        # apply translation and rotation in each step
        for i in range(steps):
            if i < accelerate_steps:
                # Acceleration phase
                incremental_translation = (np.divide(incremental_translation,
                                                     self.dt) + acc_accelerate * self.dt) * self.dt
            else:
                # Deceleration phase
                incremental_translation = (np.divide(incremental_translation,
                                                     self.dt) + acc_decelerate * self.dt) * self.dt

            # translate vertices
            vertices = positions_xzy[-1] + incremental_translation

            # calculate rotation matrix for this step
            rotation_matrix = np.array([[np.cos(rotation_angle), 0, -np.sin(rotation_angle)],
                                        [0, 1, 0],
                                        [np.sin(rotation_angle), 0, np.cos(rotation_angle)]])

            # rotate vertices
            center = vertices.mean(axis=0)
            vertices = (rotation_matrix @ (vertices - center).T).T + center

            # append vertices to positions
            positions_xzy.append(vertices)

        return positions_xzy

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

    rospy.init_node("test_move")

    client = TrajectoryClient()
    client.move_to_init_pose()

    pose_cli = PoseClient()
    t = threading.Thread(target=pose_cli.run)
    t.daemon = True
    t.start()

    client.send_cartesian_trajectory()

    try:
        while t.is_alive():
            # print("Waiting for background thread to finish")
            rospy.sleep(1)
        print("Thread finished task, exiting")
        # save log to file
        np.save('DIA/real_exp/catkin_ws/src/robot_control/results/traj_log', pose_cli.pose_log)
        np.save('DIA/real_exp/catkin_ws/src/robot_control/results/traj_desired', client.trajectory_log)
    except KeyboardInterrupt:
        print("Exit")