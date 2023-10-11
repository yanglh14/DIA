import numpy as np
import random
import pickle
import os.path as osp
import pyflex
from softgym.envs.cloth_env import ClothEnv
import copy
from copy import deepcopy


class ClothDropEnv(ClothEnv):
    def __init__(self, cached_states_path='cloth_drop_init_states.pkl', shape_type=None, **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.shape_type = shape_type
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

        assert self.action_tool.num_picker == 2  # Two drop points for this task
        self.prev_dist = None  # Should not be used until initialized

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [32, 32],
            'ClothStiff': [0.9, 1.0, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            # 'camera_params': {'default_camera':
            #                       {'pos': np.array([1.07199, 0.94942, 1.15691]),
            #                        'angle': np.array([0.633549, -0.397932, 0]),
            #                        'width': self.camera_width,
            #                        'height': self.camera_height}},
            'camera_params': {'default_camera':
                                  {'pos': np.array([1.2,0.4, 0]),
                                   'angle': np.array([1.57, -0.2, 0]),
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'flip_mesh': 0
        }
        return config

    def _get_drop_point_idx(self):
        return self._get_key_point_idx()[:2]

    def _get_vertical_pos(self, x_low, height_low):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        x = np.array(list(reversed(x)))
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)

        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = x_low
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = xx.flatten() - np.min(xx) + height_low
        return curr_pos

    def _set_to_vertical(self, x_low, height_low, height_high):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        vertical_pos = self._get_vertical_pos(x_low, height_low)
        curr_pos[:, :3] = vertical_pos
        max_height = np.max(curr_pos[:, 1])
        if max_height < height_high:
            curr_pos[:, 1] += height_high - max_height
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _get_flat_pos(self,delta_x=None, delta_z=0, delta_y=0):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        y = y - np.mean(y)
        if delta_x is None:
            delta_x = np.random.uniform(0, 0.3)
        x += delta_x
        y += delta_y
        # print(x.mean(),delta_x)
        xx, yy = np.meshgrid(x, y)
        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = xx.flatten()
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = 5e-3 + delta_z  # Set specifally for particle radius of 0.00625
        return curr_pos, delta_x

    def _set_to_flat(self):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        flat_pos = self._get_flat_pos()
        curr_pos[:, :3] = flat_pos
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def _sample_cloth_size(self):
        return np.random.randint(25, 40), np.random.randint(25, 40)

    def generate_env_variation(self, num_variations=1, vary_cloth_size=False,vary_stiffness=False):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 500  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.1  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']

            if vary_stiffness:
                config['ClothStiff'] = [np.random.uniform(0.5, 1.0), np.random.uniform(0.5, 1.0),
                                                np.random.uniform(0.5, 1.0)]
            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])

            if self.shape_type == 'platform':

                box_size = np.array([0.15, 0.02, 0.15])
                box_random_pos = self._add_box(box_size = box_size, box_pos = np.array([0.25, 0, 0]), box_quat = np.array([0., 0., 0., 1.]), random_pos=True)
                delta_x = box_random_pos - cloth_dimx * self.cloth_particle_radius / 2
                delta_z = box_size[1]
                delta_z_target = box_size[1]
                config['box_random_pos'] = box_random_pos
                config['box_height'] = box_size[1]
                config['box_size'] = box_size
                config['box_position'] = np.array([box_random_pos, 0, 0])

            elif self.shape_type == 'sphere':
                sphere_radius = 0.08
                sphere_position = np.array([0.25, 0.0, 0.0])
                sphere_random_pos = self._add_sphere(sphere_radius=sphere_radius, sphere_position=sphere_position, sphere_quat= np.array([0., 0., 0., 1.]), random_pos=True)
                delta_x = sphere_random_pos - cloth_dimx * self.cloth_particle_radius / 2
                delta_z = sphere_radius
                delta_z_target = sphere_radius

                config['sphere_random_pos'] = sphere_random_pos
                config['sphere_radius'] = sphere_radius
                config['sphere_position'] = np.array([sphere_random_pos, 0, 0])

            elif self.shape_type == 'rod':
                rod_size = np.array([0.004, 0.004, 0.2])
                rod_position = np.array([0.25, 0.1, 0.0])
                rod_random_pos = self._add_rod(rod_size=rod_size, rod_position=rod_position, rod_quat= np.array([0., 0., 0., 1.]), random_pos=True)
                delta_x = rod_random_pos - cloth_dimx * self.cloth_particle_radius / 2
                delta_z = 0.1
                delta_z_target = rod_position[1] + rod_size[1]

                config['rod_random_pos'] = rod_random_pos
                config['rod_size'] = rod_size
                config['rod_position'] = np.array([rod_random_pos, rod_position[1], rod_position[2]])

            else:
                delta_x = None
                delta_z = 0
                delta_z_target = 0

            pickpoints = self._get_drop_point_idx()[:2]  # Pick two corners of the cloth and wait until stablize

            config['target_pos'],config['delta_x'] = self._get_flat_pos(delta_x = delta_x , delta_z=delta_z_target)
            config['shape_type'] = self.shape_type

            curr_pos = pyflex.get_positions().reshape((-1, 4))
            curr_pos[:, :3] = config['target_pos']
            pyflex.set_positions(curr_pos)
            # while True:
            #     pyflex.step()
            #     pyflex.render()

            # Set the cloth to target position and wait to stablize
            for _ in range(max_wait_step):
                pyflex.step()
                pyflex.render()
                curr_vel = pyflex.get_velocities().reshape((-1, 3))
                if np.alltrue(curr_vel < stable_vel_threshold) and _ > 100:
                    break

            curr_pos = pyflex.get_positions().reshape((-1, 4))
            config['target_pos'] = curr_pos[:, :3]

            current_pos, x = self._get_flat_pos(delta_x=0, delta_z=0, delta_y=-0.3)
            curr_pos = pyflex.get_positions().reshape((-1, 4))
            curr_pos[:, :3] = current_pos
            pyflex.set_positions(curr_pos)

            # self._set_to_vertical(x_low=0, height_low=0.0, height_high=0.4 + delta_z)

            # Get height of the cloth without the gravity. With gravity, it will be longer
            p1, _, p2, _ = self._get_key_point_idx()

            curr_pos = pyflex.get_positions().reshape(-1, 4)
            # curr_pos[0] += np.random.random() * 0.001  # Add small jittering
            original_inv_mass = curr_pos[pickpoints, 3]
            curr_pos[pickpoints, 3] = 0  # Set mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            pickpoint_pos = curr_pos[pickpoints, :3]
            pyflex.set_positions(curr_pos.flatten())

            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0.05, -0.5], [0.5, 2, 0.5])
            self.action_tool.set_picker_pos(picker_pos=pickpoint_pos + np.array([0., picker_radius, 0.]))

            # Pick up the cloth and wait to stablize
            for j in range(0, max_wait_step):
                pyflex.step()
                pyflex.render()
                curr_pos = pyflex.get_positions().reshape((-1, 4))
                curr_vel = pyflex.get_velocities().reshape((-1, 3))
                if np.alltrue(curr_vel < stable_vel_threshold) and j > 300:
                    break
                curr_pos[pickpoints, :3] = pickpoint_pos
                pyflex.set_positions(curr_pos)
            curr_pos = pyflex.get_positions().reshape((-1, 4))
            curr_pos[pickpoints, 3] = original_inv_mass
            pyflex.set_positions(curr_pos.flatten())
            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def _add_rod(self, rod_size, rod_position, rod_quat, random_pos=False):
        if random_pos:
            rod_position[0] += np.random.uniform(-0.05, 0.05)
        pyflex.add_box(rod_size, rod_position, rod_quat)

        # pyflex.add_sphere(np.array(0.1), np.array([0, 0, 0]), np.array([0., 0., 0., 1.]))

        return rod_position[0]

    def _add_box(self, box_size, box_pos, box_quat, random_pos=False):

        if random_pos:
            box_pos[0] += np.random.uniform(-0.15, 0.15)
        pyflex.add_box(box_size, box_pos, box_quat)

        # pyflex.add_sphere(np.array(0.1), np.array([0, 0, 0]), np.array([0., 0., 0., 1.]))

        return box_pos[0]

    def _add_sphere(self, sphere_radius, sphere_position, sphere_quat, random_pos=False):

        if random_pos:
            sphere_position[0] += np.random.uniform(-0.15, 0.15)
        pyflex.add_sphere(sphere_radius, sphere_position, sphere_quat)

        return sphere_position[0]

    def _reset(self):
        """ Right now only use one initial state"""
        self.prev_dist = self._get_current_dist(pyflex.get_positions())
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            drop_point_pos = particle_pos[self._get_drop_point_idx(), :3]
            middle_point = np.mean(drop_point_pos, axis=0)
            self.action_tool.reset(middle_point)  # middle point is not really useful
            picker_radius = self.action_tool.picker_radius
            self.action_tool.update_picker_boundary([-0.3, 0, -0.5], [0.5, 2, 0.5])
            self.action_tool.set_picker_pos(picker_pos=drop_point_pos + np.array([0., picker_radius, 0.]))
            # self.action_tool.visualize_picker_boundary()
        if self.shape_type == 'platform':
            box_size = self.current_config['box_size']
            box_random_pos = self.current_config['box_random_pos']
            self._add_box(box_size=box_size,
                                           box_pos=np.array([box_random_pos, 0, 0]), box_quat=np.array([0., 0., 0., 1.]),
                                           random_pos=False)
        if self.shape_type == 'sphere':
            sphere_radius = self.current_config['sphere_radius']
            sphere_position = self.current_config['sphere_position']
            self._add_sphere(sphere_radius=sphere_radius, sphere_position=sphere_position,
                                                 sphere_quat=np.array([0., 0., 0., 1.]), random_pos=False)

        if self.shape_type == 'rod':
            rod_size = self.current_config['rod_size']
            rod_position = self.current_config['rod_position']
            self._add_rod(rod_size=rod_size, rod_position=rod_position, rod_quat=np.array([0., 0., 0., 1.]), random_pos=False)

        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']
        return self._get_obs()

    def _step(self, action):
        self.action_tool.step(action)
        pyflex.step()
        return

    def _get_current_dist(self, pos):
        target_pos = self.get_current_config()['target_pos']
        curr_pos = pos.reshape((-1, 4))[:, :3]
        curr_dist = np.mean(np.linalg.norm(curr_pos - target_pos, axis=1))
        return curr_dist

    def compute_reward(self, action=None, obs=None, set_prev_reward=True):
        particle_pos = pyflex.get_positions()
        curr_dist = self._get_current_dist(particle_pos)
        r = - curr_dist
        return r

    def _get_info(self):
        particle_pos = pyflex.get_positions()
        curr_dist = self._get_current_dist(particle_pos)
        performance = -curr_dist
        performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
        return {
            'performance': performance,
            'normalized_performance': (performance - performance_init) / (0. - performance_init)}
