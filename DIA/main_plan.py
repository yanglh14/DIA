import numpy as np
from DIA.planner import RandomShootingUVPickandPlacePlanner
from chester import logger
import json
import os.path as osp

import copy
import pyflex
import pickle
import multiprocessing as mp
from DIA.utils.utils import (
    downsample, transform_info, draw_planned_actions, visualize, draw_edge,
    pc_reward_model, voxelize_pointcloud, vv_to_args, set_picker_pos, cem_make_gif, configure_seed, configure_logger,draw_target_pos
)
from utils.utils_plan import *
from DIA.utils.camera_utils import get_matrix_world_to_camera, get_world_coords, get_observable_particle_index_3
from softgym.utils.visualization import save_numpy_as_gif

from DIA.dynamics import DynamicIA
from DIA.edge import Edge
import argparse
import os
import matplotlib.pyplot as plt

def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='release', help='Name of the experiment')
    parser.add_argument('--log_dir', type=str, default='data/plan/', help='Logging directory')
    parser.add_argument('--seed', type=int, default=100)

    # Env
    parser.add_argument('--env_name', type=str, default='ClothDrop', help="ClothDrop or TshirtFlatten")
    parser.add_argument('--cloth_type', type=str, default='tshirt-small', help="For 'TshirtFlatten', what types of tshir to use")
    parser.add_argument('--cached_states_path', type=str, default='dia_plan.pkl')
    parser.add_argument('--num_variations', type=int, default=20)
    parser.add_argument('--camera_name', type=str, default='default_camera')
    parser.add_argument('--down_sample_scale', type=int, default=3)
    parser.add_argument('--n_his', type=int, default=5)
    parser.add_argument('--dt', type=float, default=1. / 100.)

    parser.add_argument('--shape_type', type=str, default='None', help="Any other shape except picker: [platform, sphere, rod]")

    # Load model
    parser.add_argument('--edge_model_path', type=str, default=None,
                        help='Path to a trained edgeGNN model')
    parser.add_argument('--partial_dyn_path', type=str, default=None,
                        help='Path to a dynamics model using partial point cloud')
    parser.add_argument('--load_optim', type=bool, default=False, help='Load optimizer when resume training')

    # Planning
    parser.add_argument('--shooting_number', type=int, default=50, help='Number of sampled pick-and-place action for random shooting')
    parser.add_argument('--num_worker', type=int, default=0, help='Number of processes to generate the sampled pick-and-place actions in parallel')
    parser.add_argument('--pred_time_interval', type=int, default=5, help='Interval of timesteps between each dynamics prediction (model dt)')
    parser.add_argument('--configurations', type=list, default=[i for i in range(20)], help='List of configurations to run')
    parser.add_argument('--control_sequence_num', type=int, default=20, help='Number of pick-and-place for one smoothing trajectory')
    parser.add_argument('--delta_acc_min', type=float, default=-0.5)
    parser.add_argument('--delta_acc_max', type=float, default=0.5) # 0.1 for acc control and 0.03 for vel control

    # Other
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--voxel_size', type=float, default=0.0216, help='Pointcloud voxelization size')
    parser.add_argument('--sensor_noise', type=float, default=0, help='Artificial noise added to depth sensor')
    parser.add_argument('--gpu_num', type=int, default=1, help='# of GPUs to be used')

    # Ablation
    parser.add_argument('--fix_collision_edge', type=int, default=0, help="""
        for ablation that train without mesh edges, 
        if True, fix collision edges from the first time step during planning; 
        If False, recompute collision edge at each time step
    """)
    parser.add_argument('--use_collision_as_mesh_edge', type=int, default=0, help="""
        for ablation that train with mesh edges, but remove edge GNN at test time, 
        so it uses first-time step collision edges as the mesh edges
    """)

    args = parser.parse_args()
    return args


def prepare_policy():
    # move one of the picker to be under ground
    # not used in cloth drop env
    shape_states = pyflex.get_shape_states().reshape(-1, 14)
    shape_states[1, :3] = -1
    shape_states[1, 3:6] = -1

    # move another picker to be above the cloth
    pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
    pp = np.random.randint(len(pos))
    shape_states[0, :3] = pos[pp] + [0., 0.06, 0.]
    shape_states[0, 3:6] = pos[pp] + [0., 0.06, 0.]
    pyflex.set_shape_states(shape_states.flatten())


def create_env(args):
    from softgym.registered_env import env_arg_dict
    from softgym.registered_env import SOFTGYM_ENVS

    # create env
    env_args = copy.deepcopy(env_arg_dict[args.env_name])
    env_args['render_mode'] = 'cloth'
    env_args['observation_mode'] = 'cam_rgb'
    env_args['render'] = True
    env_args['camera_height'] = 360
    env_args['camera_width'] = 360
    env_args['camera_name'] = args.camera_name
    env_args['headless'] = True
    env_args['action_repeat'] = 1
    env_args['picker_radius'] = 0.01
    env_args['picker_threshold'] = 0.00625*100
    assert args.env_name in ['ClothDrop','TshirtFlatten']
    env_args['cached_states_path'] = args.cached_states_path
    env_args['num_variations'] = args.num_variations
    if args.env_name == 'TshirtFlatten':
        env_args['cloth_type'] = args.cloth_type
    env_args['shape_type'] = args.shape_type
    env = SOFTGYM_ENVS[args.env_name](**env_args)
    render_env_kwargs = copy.deepcopy(env_args)
    render_env_kwargs['render_mode'] = 'particle'
    render_env = SOFTGYM_ENVS[args.env_name](**render_env_kwargs)

    return env, render_env


def load_edge_model(edge_model_path, env, args):
    if edge_model_path is not None:
        edge_model_dir = osp.dirname(edge_model_path)
        edge_model_vv = json.load(open(osp.join(edge_model_dir, 'best_state.json')))
        edge_model_vv['eval'] = 1
        edge_model_vv['n_epoch'] = 1
        edge_model_vv['edge_model_path'] = edge_model_path
        edge_model_vv['shape_type'] = args.shape_type
        edge_model_args = vv_to_args(edge_model_vv)

        vcd_edge = Edge(edge_model_args, env=env)
        print('edge GNN model successfully loaded from ', edge_model_path, flush=True)
    else:
        print("no edge GNN model is loaded")
        vcd_edge = None

    return vcd_edge


def load_dynamics_model(args, env, vcd_edge):
    model_vv_dir = osp.dirname(args.partial_dyn_path)
    model_vv = json.load(open(osp.join(model_vv_dir, 'best_state.json')))

    model_vv[
        'fix_collision_edge'] = args.fix_collision_edge  # for ablation that train without mesh edges, if True, fix collision edges from the first time step during planning; If False, recompute collision edge at each time step
    model_vv[
        'use_collision_as_mesh_edge'] = args.use_collision_as_mesh_edge  # for ablation that train with mesh edges, but remove edge GNN at test time, so it uses first-time step collision edges as the mesh edges
    model_vv['train_mode'] = 'vsbl'
    model_vv['use_wandb'] = False
    model_vv['eval'] = 1
    model_vv['load_optim'] = False
    model_vv['pred_time_interval'] = args.pred_time_interval
    model_vv['cuda_idx'] = args.cuda_idx
    model_vv['partial_dyn_path'] = args.partial_dyn_path
    model_vv['shape_type'] = args.shape_type
    args = vv_to_args(model_vv)

    vcdynamics = DynamicIA(args, vcd_edge=vcd_edge, env=env)
    return vcdynamics


def get_rgbd_and_mask(env, sensor_noise):
    rgbd = env.get_rgbd(show_picker=True)
    rgb = rgbd[:, :, :3]
    depth = rgbd[:, :, 3]
    if sensor_noise > 0:
        non_cloth_mask = (depth <= 0)
        depth += np.random.normal(loc=0, scale=sensor_noise,
                                  size=(depth.shape[0], depth.shape[1]))
        depth[non_cloth_mask] = 0

    return depth.copy(), rgb, depth


def main(args):
    mp.set_start_method('forkserver', force=True)

    # Configure logger
    configure_logger(args.log_dir, args.exp_name)
    log_dir = logger.get_dir()
    # Configure seed
    configure_seed(args.seed)

    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # create env
    env, render_env = create_env(args)

    # load vcdynamics
    vcd_edge = load_edge_model(args.edge_model_path, env, args)
    vcdynamics = load_dynamics_model(args, env, vcd_edge)

    # compute camera matrix
    camera_pos, camera_angle = env.get_camera_params()
    matrix_world_to_camera = get_matrix_world_to_camera(cam_angle=camera_angle, cam_pos=camera_pos)

    # build random shooting planner
    planner = RandomShootingUVPickandPlacePlanner(
        args.shooting_number,
        dynamics=vcdynamics,
        reward_model=pc_reward_model,
        num_worker=args.num_worker,
        gpu_num=args.gpu_num,
        image_size=(env.camera_height, env.camera_width),
        matrix_world_to_camera=matrix_world_to_camera,
        dt = args.dt,
        env= env,
        args = args,
        delta_acc_range = [args.delta_acc_min, args.delta_acc_max],
        control_sequence_num = args.control_sequence_num,
    )
    # for episode_idx in args.configurations:
    #     # setup environment, ensure the same initial configuration
    #     env.reset(config_id=episode_idx)
    #     env.action_tool.update_picker_boundary([-0.3, 0.01, -0.5], [0.5, 2, 0.5])
    #
    #     config = env.get_current_config()
    #     print('config', config['ClothStiff'])

    initial_states, action_trajs, configs, all_infos, all_normalized_performance = [], [], [], [], []
    all_distance =[]
    for episode_idx in args.configurations:
        # setup environment, ensure the same initial configuration
        env.reset(config_id=episode_idx)
        env.action_tool.update_picker_boundary([-0.3, 0.0, -0.5], [0.5, 2, 0.5])

        # move one picker below the ground, set another picker randomly to a picked point / above the cloth
        # prepare_policy()

        config = env.get_current_config()
        if args.env_name == 'ClothDrop':
            cloth_xdim, cloth_ydim = config['ClothSize']
        else:
            cloth_xdim = cloth_ydim = None
        config_id = env.current_config_id
        scene_params = [env.cloth_particle_radius, cloth_xdim, cloth_ydim, config_id]

        # prepare environment and do downsample
        if args.env_name == 'ClothDrop':
            downsample_idx, downsample_x_dim, downsample_y_dim = downsample(cloth_xdim, cloth_ydim, args.down_sample_scale)
            scene_params[1] = downsample_x_dim
            scene_params[2] = downsample_y_dim
        else:
            downsample_idx = np.arange(pyflex.get_n_particles())

        initial_state = env.get_state()
        initial_states.append(initial_state), configs.append(config)

        rewards, action_traj, infos, frames = [], [], [], []
        gt_positions, gt_shape_positions, model_pred_particle_poses, model_pred_shape_poses, predicted_edges_all, predicted_performances = [], [], [], [], [], []
        actual_control_num = 0
        frames_top = []
        result_steps = {}


        flex_states, start_poses, after_poses = [env.get_state()], [], []
        obses = [env.get_image(env.camera_width, env.camera_height)]
        for control_sequence_idx in range(args.control_sequence_num):
            # prepare input data for planning
            cloth_mask, rgb, depth = get_rgbd_and_mask(env, args.sensor_noise)
            world_coordinates = get_world_coords(rgb, depth, env)[:, :, :3].reshape((-1, 3))
            pointcloud = world_coordinates[depth.flatten() > 0].astype(np.float32)
            # stop if the cloth is dragged out-of-view
            if len(pointcloud) == 0:
                print("cloth dragged out of camera view!")
                break
            voxel_pc = voxelize_pointcloud(pointcloud, args.voxel_size)
            observable_particle_indices = np.zeros(len(voxel_pc), dtype=np.int32)
            _, observable_particle_indices = get_observable_particle_index_3(voxel_pc, env.get_state()['particle_pos'].reshape(-1,4)[downsample_idx,:3])

            vel_history = np.zeros((len(observable_particle_indices), args.n_his * 3), dtype=np.float32)

            # update velocity history
            if len(gt_positions)>1:
                for i in range(min(len(gt_positions)-1,args.n_his-1)):
                    start_index = (min(control_sequence_idx, args.n_his)) * (-3) + i * 3
                    vel_history[:, start_index:start_index+3] = (gt_positions[i+1][0][observable_particle_indices] - gt_positions[i][0][observable_particle_indices]) / (args.dt * args.pred_time_interval)

                vel_history[:, -3:] = (voxel_pc -gt_positions[-1][0][observable_particle_indices]) / (args.dt * args.pred_time_interval*0.8)

            elif len(gt_positions) == 1:
                vel_history[:, -3:] = (voxel_pc -gt_positions[-1][0][observable_particle_indices]) / (args.dt * args.pred_time_interval*0.8)
            else:
                pass # vel_history is already zeros

            # current_vel = (voxel_pc - (gt_positions[-1][0][observable_particle_indices] if len(gt_positions) > 0 else voxel_pc))/(args.dt* args.pred_time_interval)
            # vel_history[:, :-3] = vel_history[:, 3:]
            # vel_history[:, -3:] = current_vel

            picker_position, picked_points = env.action_tool._get_pos()[0], [-1, -1]
            data = {
                'pointcloud': voxel_pc,
                'vel_his': vel_history,
                'picker_position': picker_position,
                'action': env.action_space.sample(),  # action will be replaced by sampled action later
                'picked_points': picked_points,
                'scene_params': scene_params,
                'partial_pc_mapped_idx': observable_particle_indices,
            }
            if args.shape_type == 'platform':
                data['box_size'] = config['box_size']
                data['box_position'] = config['box_position']
            if args.shape_type == "sphere":
                data['sphere_radius'] = config['sphere_radius']
                data['sphere_position'] = config['sphere_position']
            if args.shape_type == "rod":
                data['rod_size'] = config['rod_size']
                data['rod_position'] = config['rod_position']

            # do planning
            if control_sequence_idx <20:
                action_sequence, model_pred_particle_pos, model_pred_shape_pos, cem_info, predicted_edges, results \
                    = planner.get_action(data, control_sequence_idx=control_sequence_idx)
            else:
                action_sequence, model_pred_particle_pos, model_pred_shape_pos, cem_info, predicted_edges, results = action_sequence[1:], model_pred_particle_pos[1:], model_pred_shape_pos[1:], cem_info, predicted_edges, results

            print("config {} control sequence idx {}".format(config_id, control_sequence_idx), flush=True)

            if control_sequence_idx >= 0:
                result_steps[control_sequence_idx] = results

            # record data for plotting
            model_pred_particle_poses.append(model_pred_particle_pos)
            model_pred_shape_poses.append(model_pred_shape_pos)
            predicted_edges_all.append(predicted_edges)
            predicted_performances.append(cem_info['highest_return'])

            # decompose the large 5-step action to be small 1-step actions to execute

            if args.pred_time_interval >= 2:
                action_sequence_small = np.zeros((args.pred_time_interval, 8))
                action_sequence_small[:, :] = (action_sequence[0,:]) / args.pred_time_interval

                action_sequence_small[action_sequence_small[:, 3]>0, 3] =1
                action_sequence_small[action_sequence_small[:, 7]>0, 7] =1
                action_sequence_executed = action_sequence_small

            # execute the planned action, i.e., move the picked particle to the place location
            gt_positions.append(np.zeros((len(action_sequence_executed), len(downsample_idx), 3)))
            gt_shape_positions.append(np.zeros((len(action_sequence_executed), 2, 3)))
            for t_idx, ac in enumerate(action_sequence_executed):
                _, reward, done, info = env.step(ac, record_continuous_video=True, img_size=360)

                imgs = info['flex_env_recorded_frames']
                frames.extend(imgs)
                frames_top.append(info['image_top'])

                info.pop("flex_env_recorded_frames")
                info.pop("image_top")

                rewards.append(reward)
                action_traj.append(ac)

                gt_positions[control_sequence_idx][t_idx] = pyflex.get_positions().reshape(-1, 4)[downsample_idx, :3]
                shape_pos = pyflex.get_shape_states().reshape(-1, 14)
                for k in range(2):
                    gt_shape_positions[control_sequence_idx][t_idx][k] = shape_pos[k][:3]

            actual_control_num += 1
            infos.append(info)
            obses.append(env.get_image(env.camera_width, env.camera_height))
            flex_states.append(env.get_state())

            # early stop if the cloth is nearly smoothed
            # if info['normalized_performance'] > 0.95:
            #     break

        #####################
        # SAVE EVERYTHING
        #####################

        log_dir_episode = osp.join(log_dir, str(episode_idx))
        os.makedirs(log_dir_episode, exist_ok=True)

        #draw target pos
        target_pos = env.get_current_config()['target_pos']
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        curr_pos[:, :3] = target_pos
        pyflex.set_positions(curr_pos)
        _img = env.get_image(env.camera_width, env.camera_height)
        _imgs = [_img, _img]
        for i in range(len(_imgs)):
            _imgs[i] = draw_target_pos(_imgs[i], env.get_current_config()['target_pos'], matrix_world_to_camera[:3, :],
                                     env.camera_height, env.camera_width, env._get_key_point_idx())
        cem_make_gif([_imgs], log_dir_episode, args.env_name + '{}_target.gif'.format(episode_idx))

        # draw the planning actions & dump the data for drawing the planning actions
        draw_data = [episode_idx, flex_states, obses]
        draw_planned_actions(episode_idx, obses, matrix_world_to_camera, env.get_current_config()['target_pos'], env._get_key_point_idx(), log_dir_episode)
        with open(osp.join(log_dir_episode, '{}_draw_planned_traj.pkl'.format(episode_idx)), 'wb') as f:
            pickle.dump(draw_data, f)

        save_gif_details = True
        if save_gif_details:
            # make gif visuals of the model predictions and groundtruth rollouts
            for control_sequence_idx in range(0, actual_control_num):
                # subsample the actual rollout (since we decomposed the 5-step action to be 1-step actions)
                subsampled_gt_pos, subsampled_shape_pos = [], []
                for t in range(5):
                    subsampled_gt_pos.append(gt_positions[control_sequence_idx][t])
                    subsampled_shape_pos.append(gt_shape_positions[control_sequence_idx][t])

                frames_model = visualize(render_env, model_pred_particle_poses[control_sequence_idx], model_pred_shape_poses[control_sequence_idx],
                                         config_id, range(model_pred_particle_poses[control_sequence_idx].shape[1]))
                frames_gt = visualize(render_env, subsampled_gt_pos, subsampled_shape_pos, config_id, downsample_idx)

                # visualize the infered edge from edge GNN
                predicted_edges = predicted_edges_all[control_sequence_idx]
                frames_edge = copy.deepcopy(frames_model)
                pointcloud_pos = model_pred_particle_poses[control_sequence_idx]
                for t in range(len(frames_edge)):
                    frames_edge[t] = draw_edge(frames_edge[t], predicted_edges, matrix_world_to_camera[:3, :],
                                                      pointcloud_pos[t], env.camera_height, env.camera_width)

                # save the gif
                for name in ['gt', 'model', 'edge']:
                    _frames = np.array(eval('frames_{}'.format(name)))
                    for t in range(len(_frames)):
                        _frames[t] = draw_target_pos(_frames[t], env.get_current_config()['target_pos'], matrix_world_to_camera[:3, :],
                                             env.camera_height, env.camera_width, env._get_key_point_idx())

                    save_numpy_as_gif(_frames, osp.join(log_dir_episode,'{}-{}-{}.gif'.format(episode_idx,control_sequence_idx, name)))

                if control_sequence_idx>=0:
                    model_pred_particle_poses_all = []
                    for i in range(args.shooting_number):

                        frames_model_sample_action = visualize(render_env, result_steps[control_sequence_idx][i]['model_positions'],
                                                 model_pred_shape_poses[control_sequence_idx],
                                                 config_id, range(model_pred_particle_poses[control_sequence_idx].shape[1]))

                        _frames = np.array(frames_model_sample_action)
                        for t in range(len(_frames)):
                            _frames[t] = draw_target_pos(_frames[t], env.get_current_config()['target_pos'], matrix_world_to_camera[:3, :],
                                                 env.camera_height, env.camera_width, env._get_key_point_idx())

                        log_dir_episode_sample_action = osp.join(log_dir_episode, 'sample_action_{}'.format(control_sequence_idx))
                        os.makedirs(log_dir_episode_sample_action, exist_ok=True)
                        save_numpy_as_gif(_frames, osp.join(log_dir_episode_sample_action,'{}-{}-ActionSampling{}.gif'.format(episode_idx,control_sequence_idx , i)))

        _gt_positions = np.array(gt_positions).reshape(-1, len(downsample_idx), 3)
        _gt_shape_positions = np.array(gt_shape_positions).reshape(-1, 2, 3)
        frames_gt = visualize(render_env, _gt_positions, _gt_shape_positions, config_id, downsample_idx)
        for t in range(len(frames_gt)):
            frames_gt[t] = draw_target_pos(frames_gt[t], env.get_current_config()['target_pos'], matrix_world_to_camera[:3, :],
                                     env.camera_height, env.camera_width, env._get_key_point_idx())
        save_numpy_as_gif(np.array(frames_gt), osp.join(log_dir_episode, '{}-gt.gif'.format(episode_idx)))

        # dump traj information
        normalized_performance_traj = [info['normalized_performance'] for info in infos]
        with open(osp.join(log_dir_episode, 'normalized_performance_traj_{}.pkl'.format(episode_idx)), 'wb') as f:
            pickle.dump(normalized_performance_traj, f)

        # logging traj information
        transformed_info = transform_info([infos])
        with open(osp.join(log_dir_episode, 'transformed_info_traj_{}.pkl'.format(episode_idx)), 'wb') as f:
            pickle.dump(transformed_info, f)

        for info_name in transformed_info:

            for i in range(args.control_sequence_num):
                logger.record_tabular('reward_gt_' + 'step_{}_'.format(i) + info_name, transformed_info[info_name][0, i])
                logger.record_tabular('reward_pre_' + 'step_{}_'.format(i) + info_name, predicted_performances[i])
        logger.dump_tabular()

        # plot the performance curve using plt
        performance_gt = transformed_info['performance'][0]
        performance_pre = np.array(predicted_performances)
        plt.plot(performance_gt, label='gt')
        plt.plot(performance_pre, label='pre')
        plt.legend()
        plt.savefig(osp.join(log_dir_episode, '{}-performance.png'.format(episode_idx)))
        plt.close()

        plt.plot(_gt_shape_positions[:, 0, 0], _gt_shape_positions[:, 0, 1])
        plt.savefig(os.path.join(log_dir_episode, 'X-Z.png'))
        plt.close()

        # make gif
        for i in range(len(frames)):
            frames[i] = draw_target_pos(frames[i], env.get_current_config()['target_pos'], matrix_world_to_camera[:3, :],
                                     env.camera_height, env.camera_width,env._get_key_point_idx())

        cem_make_gif([frames], logger.get_dir(), args.env_name + '{}.gif'.format(episode_idx))

        all_normalized_performance.append(normalized_performance_traj)
        action_trajs.append(action_traj)
        all_infos.append(infos)

        # compute camera matrix
        matrix_world_to_camera_ = get_matrix_world_to_camera(cam_angle=np.array([1.57, -1.57, 0]), cam_pos=np.array([0.2,1.0, 0]))
        for i in range(len(frames_top)):
            frames_top[i] = draw_target_pos(frames_top[i], env.get_current_config()['target_pos'], matrix_world_to_camera_[:3, :],
                                     env.camera_height, env.camera_width,env._get_key_point_idx())
        cem_make_gif([frames_top], logger.get_dir(), args.env_name + '{}_top.gif'.format(episode_idx))

        all_distance.append(abs(gt_shape_positions[-1][-1,0,0] - env.get_current_config()['target_pos'][0,0]))
        print('episode {} finished'.format(episode_idx))

    # dump all data for reproducing the planned trajectory
    with open(osp.join(log_dir, 'all_normalized_performance.pkl'), 'wb') as f:
        pickle.dump(all_normalized_performance, f)
    traj_dict = {
        'initial_states': initial_states,
        'action_trajs': action_trajs,
        'configs': configs,
    }
    with open(osp.join(log_dir, 'cem_traj.pkl'), 'wb') as f:
        pickle.dump(traj_dict, f)
    all_performance = []
    for info in all_infos:
        all_performance.append(info[-1]['performance'])
    np.save(osp.join(log_dir, 'all_performance.npy'), np.array(all_performance))

    print(all_performance)
    print(np.average(all_performance))

    # end
    print('planning finished')
if __name__ == '__main__':
    main(get_default_args())
