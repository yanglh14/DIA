import numpy as np
from DIA.real_exp.catkin_ws.src.robot_control.scripts.utils.euler import quat2euler, euler2quat

class TrajectoryGenerator:

    def __init__(self):
        self.pose_init = np.array([-0.15, -0.5, 0.46, 1, 0, 0, 0])
        self.pose_init[3:] = euler2quat(np.pi, 0, 90/180*np.pi)

        self.dt = 0.01
        self.time_step = 150
        self.swing_acc_max = 1.5
        self.pull_acc_max = 0.5

        self.swing_acc = 1.5
        self.pull_acc = 0.5

    def collect_trajectory_1(self):

        cur_pose = self.pose_init
        self.target_theta = np.pi/12
        self.target_theta = 0

        bias = 0.1

        target_pose = np.zeros(7)
        target_pose[0] = cur_pose[0] - bias * np.cos(self.target_theta)
        target_pose[1] = cur_pose[1] - bias * np.sin(self.target_theta)
        target_pose[2] = 0.06
        target_pose[3:] = euler2quat(np.pi, 0, self.target_theta+np.pi/2)

        cur_rot = quat2euler(cur_pose[3:])
        target_rot = quat2euler(target_pose[3:])[2] - cur_rot[2]

        # middle state sampling

        xy_translation = 0.3
        z_ratio = 0.4

        middle_pose = target_pose.copy()
        middle_pose[0] = middle_pose[0] - xy_translation * np.cos(target_rot)
        middle_pose[1] = middle_pose[1] - xy_translation * np.sin(target_rot)
        middle_pose[2] = cur_pose[2] + z_ratio * (target_pose[2] - cur_pose[2])

        trajectory_s2m = self._generate_trajectory_1(cur_pose, middle_pose, self.swing_acc_max, self.dt)
        trajectory_m2e = self._generate_trajectory_1(middle_pose, target_pose, self.pull_acc_max, self.dt)

        trajectory = np.concatenate((trajectory_s2m, trajectory_m2e[1:]), axis=0)

        return trajectory

    def _generate_trajectory_1(self, current_pos, target_pos, acc_max, dt):
        translation = (target_pos[:3] - current_pos[:3])
        rotation = np.array(quat2euler(target_pos[3:])) - np.array(quat2euler(current_pos[3:]))

        # Calculate the number of time steps
        _time_steps = max(np.sqrt(4 * np.abs(translation) / acc_max) / dt)
        time_steps = np.ceil(_time_steps).max().astype(int)
        print('time_steps', time_steps)
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
            _pos[3:] = euler2quat( *(rot_steps * i + quat2euler(current_pos[3:])) )

            positions_xyzq.append(_pos)

        return np.array(positions_xyzq)

    def _collect_trajectory_2(self, current_picker_position, target_picker_position):

        """ Policy for collecting data - random sampling"""

        middle_position_xy_translation = 0.3
        middle_position_z_ratio = 0.4

        norm_direction = np.array([target_picker_position[1, 2] - target_picker_position[0, 2],
                                   target_picker_position[0, 0] - target_picker_position[1, 0]]) / \
                         np.linalg.norm(np.array([target_picker_position[1, 2] - target_picker_position[0, 2],
                                                  target_picker_position[0, 0] - target_picker_position[1, 0]]))
        middle_state = target_picker_position.copy()
        middle_state[:, [0, 2]] = target_picker_position[:, [0, 2]] + middle_position_xy_translation * norm_direction
        middle_state[:, 1] = current_picker_position[:, 1] + middle_position_z_ratio * (
                target_picker_position[:, 1] - current_picker_position[:, 1])

        trajectory_start_to_middle = self._trajectory_generation_2(current_picker_position, middle_state,
                                                                 self.swing_acc, self.dt)

        trajectory_middle_to_target = self._trajectory_generation_2(middle_state, target_picker_position,
                                                                  self.pull_acc, self.dt)

        # cat trajectory_xy and trajectory_z
        trajectory = np.concatenate((trajectory_start_to_middle, trajectory_middle_to_target[1:]), axis=0)
        trajectory = trajectory.reshape(trajectory.shape[0], -1)

        action_list = []
        for step in range(1, trajectory.shape[0]):
            action = np.ones(8)
            action[:3], action[4:7] = trajectory[step, :3] - trajectory[step - 1, :3], trajectory[step,
                                                                                       3:6] - trajectory[step - 1, 3:6]
            action_list.append(action)

        action_list = np.array(action_list)
        action_list[:, [3, 7]] = 1
        return action_list

    def _trajectory_generation_2(self, current_picker_position, target_picker_position, acc_max, dt):

        """ Policy for trajectory generation based on current and target_picker_position"""

        # select column 1 and 3 in current_picker_position and target_picker_position
        initial_vertices_xy = current_picker_position[:, [0, 2]]
        final_vertices_xy = target_picker_position[:, [0, 2]]

        # calculate angle of rotation from initial to final segment in xy plane
        angle = np.arctan2(final_vertices_xy[1, 1] - final_vertices_xy[0, 1],
                           final_vertices_xy[1, 0] - final_vertices_xy[0, 0]) - \
                np.arctan2(initial_vertices_xy[1, 1] - initial_vertices_xy[0, 1],
                           initial_vertices_xy[1, 0] - initial_vertices_xy[0, 0])

        # translation vector: difference between final and initial centers
        translation = (target_picker_position.mean(axis=0) - current_picker_position.mean(axis=0))

        _time_steps = max(np.sqrt(4 * np.abs(translation) / acc_max) / dt)
        steps = np.ceil(_time_steps).max().astype(int)

        # calculate angle of rotation for each step
        rot_steps = angle / steps

        accel_steps = steps // 2
        decel_steps = steps - accel_steps

        v_max = translation * 2 / (steps * dt)
        accelerate = v_max / (accel_steps * dt)
        decelerate = -v_max / (decel_steps * dt)

        # calculate incremental translation
        incremental_translation = [0, 0, 0]

        # initialize list of vertex positions
        positions_xzy = [current_picker_position]

        # apply translation and rotation in each step
        for i in range(steps):
            if i < accel_steps:
                # Acceleration phase
                incremental_translation = (np.divide(incremental_translation,
                                                     self.dt) + accelerate * self.dt) * self.dt
            else:
                # Deceleration phase
                incremental_translation = (np.divide(incremental_translation,
                                                     self.dt) + decelerate * self.dt) * self.dt

            # translate vertices
            vertices = positions_xzy[-1] + incremental_translation

            # calculate rotation matrix for this step
            rotation_matrix = np.array([[np.cos(rot_steps), 0, -np.sin(rot_steps)],
                                        [0, 1, 0],
                                        [np.sin(rot_steps), 0, np.cos(rot_steps)]])

            # rotate vertices
            center = vertices.mean(axis=0)
            vertices = (rotation_matrix @ (vertices - center).T).T + center

            # append vertices to positions
            positions_xzy.append(vertices)

        return positions_xzy


if __name__ == '__main__':

    generator = TrajectoryGenerator()
    trajectory_1 = generator.collect_trajectory_1()

    current_picker_position = np.array([[0, 0.4, -0.1], [0, 0.4, 0.1]])

    target_picker_position = np.array([[0.1, 0, -0.1], [0.1, 0, 0.1]])

    trajectory_2 = generator._collect_trajectory_2(current_picker_position, target_picker_position)
    print(trajectory_2.shape,)

