import numpy as np
from multiprocessing import pool
import copy
from DIA.utils.camera_utils import project_to_image, get_target_pos


class RandomShootingUVPickandPlacePlanner():

    def __init__(self, num_pick,
                    dynamics, reward_model, num_worker=10, gpu_num=1,
                    image_size=None, normalize_info=None,
                    matrix_world_to_camera=np.identity(4),
                    use_pred_rwd=False,
                    delta_acc_range=[0, 1], dt=0.01, env=None, args=None,control_sequence_num=20):
        """
        Random Shooting planner.
        """

        self.normalize_info = normalize_info  # Used for robot experiments to denormalize before action clipping
        self.num_pick = num_pick
        self.reward_model = reward_model
        self.dynamics = dynamics
        self.gpu_num = gpu_num
        self.use_pred_rwd = use_pred_rwd

        if num_worker > 0:
            self.pool = pool.Pool(processes=num_worker)
        self.num_worker = num_worker
        self.matrix_world_to_camera = matrix_world_to_camera
        self.image_size = image_size

        self.delta_acc_range = delta_acc_range
        self.dt = dt
        self.env = env
        self.args = args
        self.control_sequence_num = control_sequence_num
    def project_3d(self, pos):
        return project_to_image(self.matrix_world_to_camera, pos, self.image_size[0], self.image_size[1])

    def get_action(self, init_data, control_sequence_idx = 0,  robot_exp=False, m_name='vsbl'):
        """
        check_mask: Used to filter out place points that are on the cloth.
        init_data should be a list that include:
            ['pointcloud', 'velocities', 'picker_position', 'action', 'picked_points', 'scene_params', 'observable_particle_indices]
            note: require position, velocity to be already downsampled

        """
        data = init_data.copy()
        data['picked_points'] = [-1, -1]

        # add a no-op action
        sampling_num = self.num_pick
        pointcloud = copy.deepcopy(data['pointcloud'])

        # paralleled version of generating action sequences
        if robot_exp:

            raise NotImplementedError

        else:  # simulation planning
            if control_sequence_idx==0:
                self.tool_state = np.zeros(6)

            interval = (self.delta_acc_range[1]-self.delta_acc_range[0])/self.num_pick
            sampling_noise = [self.delta_acc_range[0] + i * interval for i in range(self.num_pick)]

            params = [
                (control_sequence_idx, self.control_sequence_num, sampling_noise[i], self.dt, self.env, self.args.pred_time_interval, self.tool_state.copy())
                for i in range(self.num_pick)
            ]

            if self.num_worker > 0:
                results = self.pool.map(_parallel_generate_actions_v4, params)
            else:
                # results = [_parallel_generate_actions_v3(param) for param in params]
                if self.args.shape_type == 'rod':
                    results = [_parallel_generate_actions_rod(param) for param in params]
                elif self.args.shape_type == 'sphere':
                    results = [_parallel_generate_actions_sphere(param) for param in params]
                elif self.args.shape_type == 'platform':
                    results = [_parallel_generate_actions_v4_pick_drop(param) for param in params]
            actions = np.array(results)

        # parallely rollout the dynamics model with the sampled action seqeunces
        data_cpy = copy.deepcopy(data)
        if self.num_worker > 0:
            job_each_gpu = sampling_num // self.gpu_num
            params = []
            for i in range(sampling_num):

                gpu_id = i // job_each_gpu if i < self.gpu_num * job_each_gpu else i % self.gpu_num
                params.append(
                    dict(
                        model_input_data=copy.deepcopy(data_cpy), actions=actions[i], m_name=m_name,
                        reward_model=self.reward_model, cuda_idx=gpu_id, robot_exp=robot_exp,
                    )
                )
            results = self.pool.map(self.dynamics.rollout, params, chunksize=max(1, sampling_num // self.num_worker))
            returns = [x['final_ret'] for x in results]
        else: # sequentially rollout each sampled action trajectory
            returns, results = [], []
            for i in range(sampling_num):
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
        self.tool_state[:3] = actions[highest_return_idx, 0, :3]

        ret_info['highest_return_idx'] = highest_return_idx
        ret_info['highest_return'] = returns[highest_return_idx]
        action_seq = actions[highest_return_idx]

        model_predict_particle_positions = results[highest_return_idx]['model_positions']
        model_predict_shape_positions = results[highest_return_idx]['shape_positions']
        predicted_edges = results[highest_return_idx]['mesh_edges']
        print('highest_return_idx', highest_return_idx)
        return action_seq, model_predict_particle_positions, model_predict_shape_positions, ret_info, predicted_edges, results


def pos_in_image(after_pos, matrix_world_to_camera, image_size):
    euv = project_to_image(matrix_world_to_camera, after_pos.reshape((1, 3)), image_size[0], image_size[1])
    u, v = euv[0][0], euv[1][0]
    if u >= 0 and u < image_size[1] and v >= 0 and v < image_size[0]:
        return True
    else:
        return False

def _parallel_generate_actions_v4(args):
    # sample action for v4 pick and drop
    control_sequence_idx, control_sequence_num, delta_acc_range, dt, env, pred_time_interval,state = args
    acc_delta_value = np.random.uniform(
        delta_acc_range[0],
        delta_acc_range[1], size=control_sequence_num)

    action_list = []

    for step in range(control_sequence_idx, control_sequence_num):

        if step < control_sequence_num/ 4:
            acc_direction = np.array([6, -1.6, 0], dtype=np.float32)

        elif step < control_sequence_num / 2:
            acc_direction = np.array([-6, -1.6, 0], dtype=np.float32)

        elif step < control_sequence_num * 3 / 4:
            acc_direction = np.array([-3, 1.6, 0], dtype=np.float32)

        else:
            acc_direction = np.array([3, 1.6, 0], dtype=np.float32)

        acc_direction[0] = acc_direction[0] + acc_delta_value[step]

        state[3:6] = acc_direction * dt * pred_time_interval

        state[0:3] += state[3:6] * dt * pred_time_interval

        action = np.zeros_like(env.action_space.sample())
        action[:3], action[4:7] = state[0:3], state[0:3]
        if not step > control_sequence_num * 0.75:
            action[7], action[3] = 1, 1
        else:
            action[7], action[3] = 0, 0
        action_list.append(action)

    return action_list

def _parallel_generate_actions_v4_pick_drop(args):
    # in this sampling, only first action change
    control_sequence_idx, control_sequence_num, sampling_noise, dt, env, pred_time_interval,state = args

    action_list = []

    for step in range(control_sequence_idx, control_sequence_num):

        if step < control_sequence_num/ 4:
            acc_direction = np.array([6, -1.6, 0], dtype=np.float32)

        elif step < control_sequence_num / 2:
            acc_direction = np.array([-6, -1.6, 0], dtype=np.float32)

        elif step < (control_sequence_num * 3 / 4) - control_sequence_num /8:
            acc_direction = np.array([-5 - sampling_noise, 1.6, 0], dtype=np.float32)

        elif step < (control_sequence_num * 3 / 4):
            acc_direction = np.array([7.5 - sampling_noise, 1.6, 0], dtype=np.float32)

        # elif step < (control_sequence_num * 3 / 4):
        #     acc_direction = np.array([-3, 1.6, 0], dtype=np.float32)

        else:
            acc_direction = np.array([3, 1.6, 0], dtype=np.float32)

        if step == control_sequence_idx:
            acc_direction[0] = acc_direction[0] + sampling_noise

        state[3:6] = acc_direction * dt * pred_time_interval

        state[0:3] += state[3:6] * dt * pred_time_interval

        action = np.zeros_like(env.action_space.sample())
        action[:3], action[4:7] = state[0:3], state[0:3]
        action[7], action[3] = 1, 1

        if not step >= control_sequence_num * 0.75:
            action[7], action[3] = 1, 1
        else:
            action[:]= 0
        action_list.append(action)

    return action_list

def _parallel_generate_actions_sphere(args):
    """ sampling actions for sphere """

    control_sequence_idx, control_sequence_num, sampling_noise, dt, env, pred_time_interval,state = args

    action_list = []

    for step in range(control_sequence_idx, control_sequence_num):

        if step <= control_sequence_num*0.8/ 4:
            acc_direction = np.array([6, -1.6, 0], dtype=np.float32)

        elif step <= control_sequence_num*0.8 / 2:
            acc_direction = np.array([-6, -1.6, 0], dtype=np.float32)

        elif step <= (control_sequence_num*0.8 * 3 / 4):
            acc_direction = np.array([-3, 1.6, 0], dtype=np.float32)

        elif step <= control_sequence_num*0.8:
            acc_direction = np.array([3, 1.6, 0], dtype=np.float32)

        else:
            acc_direction = np.array([0, 0, 0], dtype=np.float32)

        if step == control_sequence_idx:
            acc_direction[0] = acc_direction[0] + sampling_noise

        state[3:6] = acc_direction * dt * pred_time_interval

        state[0:3] += state[3:6] * dt * pred_time_interval

        action = np.zeros_like(env.action_space.sample())
        action[:3], action[4:7] = state[0:3], state[0:3]

        if not step >= control_sequence_num * 0.8* 3 / 4:
            action[7], action[3] = 1, 1
        else:
            action[:]= 0
        action_list.append(action)

    return action_list

def _parallel_generate_actions_rod(args):
    """ sampling actions for rod """

    control_sequence_idx, control_sequence_num, sampling_noise, dt, env, pred_time_interval,state = args

    action_list = []

    for step in range(control_sequence_idx, control_sequence_num):

        if step <= control_sequence_num*0.8/ 4:
            acc_direction = np.array([6, -1.6, 0], dtype=np.float32)

        elif step <= control_sequence_num*0.8 / 2:
            acc_direction = np.array([-6, -1.6, 0], dtype=np.float32)

        elif step <= (control_sequence_num*0.8 * 3 / 4):
            acc_direction = np.array([-3, 1.6, 0], dtype=np.float32)

        elif step <= control_sequence_num*0.8:
            acc_direction = np.array([3, 1.6, 0], dtype=np.float32)

        else:
            acc_direction = np.array([0, 0, 0], dtype=np.float32)

        if step == control_sequence_idx:
            acc_direction[0] = acc_direction[0] + sampling_noise

        state[3:6] = acc_direction * dt * pred_time_interval

        state[0:3] += state[3:6] * dt * pred_time_interval

        action = np.zeros_like(env.action_space.sample())
        action[:3], action[4:7] = state[0:3], state[0:3]

        if not step >= control_sequence_num * 0.8:
            action[7], action[3] = 1, 1
        else:
            action[:]= 0
        action_list.append(action)

    return action_list