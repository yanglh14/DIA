import os
import numpy as np

import pyflex

from DIA.utils.utils import downsample, load_data, load_data_list, store_h5_data, voxelize_pointcloud, pc_reward_model, draw_target_pos
from DIA.utils.camera_utils import get_observable_particle_index, get_observable_particle_index_old, get_world_coords, get_observable_particle_index_3, get_matrix_world_to_camera
from softgym.utils.visualization import save_numpy_as_gif

import matplotlib.pyplot as plt

class DataCollector(object):
    def __init__(self, args, phase, env):

        self.args = args
        self.env = env
        self.dt = args.dt
        self.data_names = None # just load and save everything
        ratio = self.args.train_valid_ratio

        if self.args.dataf is not None:
            self.data_dir = os.path.join(self.args.dataf, phase)
            os.system('mkdir -p ' + self.data_dir)

        if phase == 'train':
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase == 'valid':
            self.n_rollout = int(self.args.n_rollout - int(self.args.n_rollout * ratio))
        else:
            raise AssertionError("Unknown phase")

    def gen_dataset(self):
        np.random.seed(0)
        rollout_idx = 0
        while rollout_idx < self.n_rollout:
            print("{} / {}".format(rollout_idx, self.n_rollout))
            rollout_dir = os.path.join(self.data_dir, str(rollout_idx))
            os.system('mkdir -p ' + rollout_dir)
            self.env.reset()
            prev_data = self.get_curr_env_data()  # Get new picker position

            if self.args.gen_gif:
                frames_rgb, frames_depth = [prev_data['rgb']], [prev_data['depth']]
            current_config = self.env.get_current_config()

            actions = self._collect_trajectory(prev_data['picker_position'], current_config[
                'target_picker_pos'])  # Get actions for the baseline and platform, len=99

            picker_position_list = []
            for j in range(1, self.args.time_step):

                self.env.action_tool.update_picker_boundary([-0.3, 0, -0.5], [1, 2, 0.5])
                if not self._data_test(prev_data):
                    # raise error
                    raise Exception('_data_test failed: Number of point cloud too small')

                action = actions[j - 1]

                self.env.step(action)
                curr_data = self.get_curr_env_data()
                picker_position_list.append(curr_data['picker_position'])

                prev_data['velocities'] = (curr_data['positions'] - prev_data['positions']) / self.dt
                prev_data['action'] = action
                store_h5_data(self.data_names, prev_data, os.path.join(rollout_dir, str(j - 1) + '.h5'))
                prev_data = curr_data
                if self.args.gen_gif:
                    frames_rgb.append(prev_data['rgb'])
                    frames_depth.append(prev_data['depth'])

            if self.args.gen_gif:
                camera_pos, camera_angle = self.env.get_camera_params()
                matrix_world_to_camera = get_matrix_world_to_camera(cam_angle=camera_angle, cam_pos=camera_pos)

                frames_rgb = np.array(np.array(frames_rgb) * 255).clip(0., 255.)
                for t in range(len(frames_rgb)):
                    frames_rgb[t] = draw_target_pos(frames_rgb[t], self.env.get_current_config()['target_pos'],
                                                    matrix_world_to_camera[:3, :],
                                                    self.env.camera_height, self.env.camera_width,
                                                    self.env._get_key_point_idx())

                save_numpy_as_gif(frames_rgb, os.path.join(rollout_dir, 'rgb.gif'))
                save_numpy_as_gif(np.array(frames_depth) * 255., os.path.join(rollout_dir, 'depth.gif'))

                # plot the picker position in 3 axis
                picker_position_list = np.array(picker_position_list)

                fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True,
                                                    figsize=(12, 6))
                ax0.plot(picker_position_list[:, 0, 0])
                ax1.plot(picker_position_list[:, 0, 1])
                ax2.plot(picker_position_list[:, 0, 2])
                plt.savefig(os.path.join(rollout_dir, 'picker_position.png'))
                plt.close(fig)
                plt.plot(picker_position_list[:, 0, 0], picker_position_list[:, 0, 1])
                plt.savefig(os.path.join(rollout_dir, 'X-Z.png'))
                plt.close(fig)

            # the last step has no action, and is not used in training
            prev_data['action'], prev_data['velocities'] = 0, 0
            store_h5_data(self.data_names, prev_data, os.path.join(rollout_dir, str(self.args.time_step - 1) + '.h5'))
            rollout_idx += 1

    def get_curr_env_data(self):
        # Env info that does not change within one episode
        config = self.env.get_current_config()
        cloth_xdim, cloth_ydim = config['ClothSize']
        config_id = self.env.current_config_id
        scene_params = [self.env.cloth_particle_radius, cloth_xdim, cloth_ydim, config_id]

        downsample_idx, downsample_x_dim, downsample_y_dim = downsample(cloth_xdim, cloth_ydim, self.args.down_sample_scale)
        scene_params[1], scene_params[2] = downsample_x_dim, downsample_y_dim

        position = pyflex.get_positions().reshape(-1, 4)[:, :3]
        picker_position = self.env.action_tool.get_picker_pos()

        # Cloth and picker information
        # Get partially observed particle index
        rgbd = self.env.get_rgbd(show_picker=True)
        rgb, depth = rgbd[:, :, :3], rgbd[:, :, 3]

        world_coordinates = get_world_coords(rgb, depth, self.env, position)

        # Old way of getting observable index
        downsample_observable_idx = get_observable_particle_index_old(world_coordinates, position[downsample_idx], rgb, depth)
        # TODO Try new way of getting observable index
        observable_idx = get_observable_particle_index(world_coordinates, position, rgb, depth)
        # all_idx = np.zeros(shape=(len(position)), dtype=np.int)
        # all_idx[observable_idx] = 1
        # downsample_observable_idx = np.where(all_idx[downsample_idx] > 0)[0]

        world_coords = world_coordinates[:, :, :3].reshape((-1, 3))
        pointcloud = world_coords[depth.flatten() > 0]

        ret = {'positions': position.astype(np.float32),
               'picker_position': picker_position,
               'scene_params': scene_params,
               'downsample_idx': downsample_idx,
               'downsample_observable_idx': downsample_observable_idx,
               'observable_idx': observable_idx,
               'pointcloud': pointcloud.astype(np.float32)}

        current_config = self.env.get_current_config()
        ret['target_pos'] = current_config['target_pos']

        if self.args.gen_gif:
            ret['rgb'], ret['depth'] = rgb, depth

        if self.env.env_shape != None:
            ret['shape_size'] = current_config['shape_size']
            ret['shape_pos'] = current_config['shape_pos']
            ret['shape_quat'] = current_config['shape_quat']

        return ret

    def _collect_trajectory(self, current_picker_position, target_picker_position):

        """ Policy for collecting data - random sampling"""

        middle_position_step_ratio = np.random.uniform(0.3, 0.7)
        middle_position_xy_translation = np.random.uniform(0.1, 0.3)
        middle_position_z_ratio = np.random.uniform(0.2, 0.5)

        norm_direction = np.array([target_picker_position[1, 2] - target_picker_position[0, 2],
                                   target_picker_position[0, 0] - target_picker_position[1, 0]]) / \
                         np.linalg.norm(np.array([target_picker_position[1, 2] - target_picker_position[0, 2],
                                                  target_picker_position[0, 0] - target_picker_position[1, 0]]))
        middle_state = target_picker_position.copy()
        middle_state[:, [0, 2]] = target_picker_position[:, [0, 2]] + middle_position_xy_translation * norm_direction
        middle_state[:, 1] = current_picker_position[:, 1] + middle_position_z_ratio * (
                    target_picker_position[:, 1] - current_picker_position[:, 1])

        trajectory_start_to_middle = self._trajectory_generation(current_picker_position, middle_state,
                                                              int(self.args.time_step * middle_position_step_ratio))

        trajectory_middle_to_target = self._trajectory_generation(middle_state, target_picker_position,
                                                            self.args.time_step - int(
                                                                self.args.time_step * middle_position_step_ratio))

        # cat trajectory_xy and trajectory_z
        trajectory = np.concatenate((trajectory_start_to_middle, trajectory_middle_to_target[1:]), axis=0)
        trajectory = trajectory.reshape(trajectory.shape[0], -1)

        action_list = []
        for step in range(1, self.args.time_step):
            action = np.ones_like(self.env.action_space.sample(), dtype=np.float32)
            action[:3], action[4:7] = trajectory[step, :3] - trajectory[step - 1, :3], trajectory[step,
                                                                                       3:6] - trajectory[step - 1, 3:6]
            action_list.append(action)

        return action_list

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

    def _data_test(self, data):
        """ Filter out cases where cloth is moved out of the view or when number of voxelized particles is larger than number of partial particles"""
        pointcloud = data['pointcloud']
        if len(pointcloud.shape) != 2 or len(pointcloud) < 100:
            print('_data_test failed: Number of point cloud too small')
            return False
        return True
