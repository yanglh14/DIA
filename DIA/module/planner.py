import numpy as np
from multiprocessing import pool
import copy
from DIA.utils.camera_utils import project_to_image, get_target_pos


class MPCPlanner():

    def __init__(self,
                    dynamics, reward_model, normalize_info=None,
                    matrix_world_to_camera=np.identity(4),
                    use_pred_rwd=False, env=None, args=None):
        """
        Random Shooting planner.
        """

        self.normalize_info = normalize_info  # Used for robot experiments to denormalize before action clipping
        self.shooting_number = args.shooting_number
        self.reward_model = reward_model
        self.dynamics = dynamics
        self.gpu_num = args.gpu_num
        self.use_pred_rwd = use_pred_rwd

        num_worker = args.num_worker
        if num_worker > 0:
            self.pool = pool.Pool(processes=num_worker)

        self.num_worker = num_worker
        self.matrix_world_to_camera = matrix_world_to_camera
        self.image_size = (env.camera_height, env.camera_width)

        self.dt = args.dataset.dt
        self.env = env
        self.args = args
        self.sequence_steps = args.sequence_steps
        self.pred_time_interval = args.dataset.pred_time_interval

    def init_traj(self, data, episode_idx, m_name='vsbl'):

        picker_position = self.env.action_tool._get_pos()[0]
        self.target_position = self.env.get_current_config()['target_picker_pos']
        actions_sampled, step_mid_sampled, middle_state_sampled = [], [], []
        for _ in range(10):
            actions, step_mid, middle_state = _collect_trajectory((picker_position, self.target_position, self.dt, self.pred_time_interval, self.sequence_steps))
            actions_sampled.append(actions)
            step_mid_sampled.append(step_mid)
            middle_state_sampled.append(middle_state)

        data_cpy = copy.deepcopy(data)

        returns, frames_var, frames_top_var = [], [], []
        for i in range(len(actions_sampled)):

            frames, frames_top, reward = rollout_gt(self.env, episode_idx, actions_sampled[i], self.pred_time_interval)
            returns.append(reward)
            frames_var.append(frames)
            frames_top_var.append(frames_top)

        highest_return_idx = np.argmax(returns)
        self.actions = actions_sampled[highest_return_idx]
        self.step_mid = step_mid_sampled[highest_return_idx]

        # How to evaluate this middle state? add noise to the dynamics model..  or add noise to the action sequence?
        return frames_var, frames_top_var, returns

    def update_traj(self, actions, control_sequence_idx):
        self.actions[control_sequence_idx:] = actions

    def get_action(self, init_data, control_sequence_idx=0,  robot_exp=False, m_name='vsbl'):
        """
        check_mask: Used to filter out place points that are on the cloth.
        init_data should be a list that include:
            ['pointcloud', 'velocities', 'picker_position', 'action', 'picked_points', 'scene_params', 'observable_particle_indices]
            note: require position, velocity to be already downsampled

        """
        data = init_data.copy()

        # maybe stop here or continue?

        # paralleled version of generating action sequences
        if robot_exp:
            raise NotImplementedError
        else:  # simulation planning

            if control_sequence_idx < self.step_mid:
                actions_swing = self.actions[control_sequence_idx:self.step_mid]

                ## add noise to actions_swing
                noise_ratio = np.random.normal(0, 0.2, [self.shooting_number,3])
                delta_action_list = [actions_swing[:, :3] * noise_ratio[i] for i in range(self.shooting_number)]

                # extend one dimension
                actions_swing = np.expand_dims(actions_swing, axis=0).repeat(self.shooting_number, axis=0)

                for i in range(self.shooting_number):
                    actions_swing[i,:,:3] = actions_swing[i,:,:3] + delta_action_list[i]
                    actions_swing[i,:,4:7] = actions_swing[i,:,4:7] + delta_action_list[i]

                assumed_mid_pos_1 = data['picker_position'][0] + np.sum(actions_swing[:,:, :3], axis=1)
                assumed_mid_pos_2 = data['picker_position'][1] + np.sum(actions_swing[:,:, 4:7], axis=1)
                assumed_mid_pos = np.concatenate((assumed_mid_pos_1, assumed_mid_pos_2), axis=1)

                actions_pull_list = []
                for i in range(self.shooting_number):
                    traj_pull = _trajectory_generation(assumed_mid_pos[i].reshape(-1, 3), self.target_position,
                                                          self.sequence_steps-self.step_mid, self.dt*self.pred_time_interval)

                    actions_pull = []
                    for step in range(self.sequence_steps-self.step_mid):
                        action = np.ones(8, dtype=np.float32)
                        action[:3], action[4:7] = traj_pull[step+1][0] - traj_pull[step][0], traj_pull[step+1][1] - traj_pull[step][1]
                        actions_pull.append(action)

                    actions_pull_list.append(actions_pull)
                actions = np.concatenate((actions_swing, actions_pull_list), axis=1)

            else:
                actions = self.actions[control_sequence_idx:]
                actions = np.expand_dims(actions, axis=0).repeat(self.shooting_number, axis=0)

        # parallely rollout the dynamics model with the sampled action seqeunces
        data_cpy = copy.deepcopy(data)
        if self.num_worker > 0:
            job_each_gpu = self.shooting_number // self.gpu_num
            params = []
            for i in range(self.shooting_number):

                gpu_id = i // job_each_gpu if i < self.gpu_num * job_each_gpu else i % self.gpu_num
                params.append(
                    dict(
                        model_input_data=copy.deepcopy(data_cpy), actions=actions[i], m_name=m_name,
                        reward_model=self.reward_model, cuda_idx=gpu_id, robot_exp=robot_exp,
                    )
                )
            results = self.pool.map(self.dynamics.rollout, params, chunksize=max(1, self.shooting_number // self.num_worker))
            returns = [x['final_ret'] for x in results]
        else: # sequentially rollout each sampled action trajectory
            returns, results = [], []
            for i in range(self.shooting_number):
                assert actions[i].shape[-1] == 8
                res = self.dynamics.rollout(
                    dict(
                        model_input_data=copy.deepcopy(data_cpy), actions=actions[i], m_name=m_name,
                        reward_model=self.reward_model, cuda_idx=0, robot_exp=robot_exp,
                    )
                )
                results.append(res), returns.append(res['final_ret'])

        ret_info = {}
        highest_return_idx = np.argmax(returns)

        ret_info['highest_return_idx'] = highest_return_idx
        ret_info['highest_return'] = returns[highest_return_idx]
        action_seq = actions[highest_return_idx]

        model_predict_positions = results[highest_return_idx]['model_positions']
        model_predict_shape_positions = results[highest_return_idx]['shape_positions']
        predicted_edges = results[highest_return_idx]['mesh_edges']
        print('highest_return_idx', highest_return_idx)

        self.update_traj(action_seq, control_sequence_idx=control_sequence_idx)

        return action_seq, model_predict_positions, model_predict_shape_positions, ret_info, predicted_edges, results


def pos_in_image(after_pos, matrix_world_to_camera, image_size):
    euv = project_to_image(matrix_world_to_camera, after_pos.reshape((1, 3)), image_size[0], image_size[1])
    u, v = euv[0][0], euv[1][0]
    if u >= 0 and u < image_size[1] and v >= 0 and v < image_size[0]:
        return True
    else:
        return False

def project_3d(self, pos):
    return project_to_image(self.matrix_world_to_camera, pos, self.image_size[0], self.image_size[1])

def _collect_trajectory(args):

    """ Policy for collecting data - random sampling"""

    current_picker_position, target_picker_position, dt, pred_time_interval, time_step = args

    dt = dt * pred_time_interval

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

    trajectory_start_to_middle = _trajectory_generation(current_picker_position, middle_state,
                                                          int(time_step * middle_position_step_ratio), dt)

    trajectory_middle_to_target = _trajectory_generation(middle_state, target_picker_position,
                                                        time_step - int(
                                                            time_step * middle_position_step_ratio), dt)

    # cat trajectory_xy and trajectory_z
    trajectory = np.concatenate((trajectory_start_to_middle, trajectory_middle_to_target[1:]), axis=0)
    trajectory = trajectory.reshape(trajectory.shape[0], -1)

    action_list = []
    for step in range(time_step):
        action = np.ones(8, dtype=np.float32)
        action[:3], action[4:7] = trajectory[step+1, :3] - trajectory[step , :3], trajectory[step+1,
                                                                                   3:6] - trajectory[step , 3:6]
        action_list.append(action)

    return np.array(action_list), int(time_step * middle_position_step_ratio), middle_state

def _trajectory_generation(current_picker_position, target_picker_position, time_steps, dt):

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

    v_max = translation * 2 / ((accelerate_steps + decelerate_steps) * dt)
    acc_accelerate = v_max / (accelerate_steps * dt)
    acc_decelerate = -v_max / (decelerate_steps * dt)

    # apply translation and rotation in each step
    for i in range(steps):
        if i < accelerate_steps:
            # Acceleration phase
            incremental_translation = (np.divide(incremental_translation,
                                                 dt) + acc_accelerate * dt) * dt
        else:
            # Deceleration phase
            incremental_translation = (np.divide(incremental_translation,
                                                 dt) + acc_decelerate * dt) * dt

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

def rollout_gt(env, episode_idx, actions, pred_time_interval):
    # rollout the env in simulation
    env.reset(config_id=episode_idx)
    actions_executed = []
    for action in actions:
        action_sequence_small = np.zeros((pred_time_interval, 8))
        action_sequence_small[:, :] = (action[:]) / pred_time_interval

        action_sequence_small[action_sequence_small[:, 3] > 0, 3] = 1
        action_sequence_small[action_sequence_small[:, 7] > 0, 7] = 1
        actions_executed.extend(action_sequence_small)

    frames, frames_top = [], []
    for action in actions_executed:
        _, reward, done, info = env.step(action, record_continuous_video=True, img_size=360)

        frames.extend(info['flex_env_recorded_frames'])
        frames_top.append(info['image_top'])

    return frames, frames_top, reward