from DIA.module.planner import MPCPlanner
from chester import logger
import json

import pickle
import multiprocessing as mp

from DIA.utils.plot_utils import *

from DIA.module.dynamics import DynamicIA
from DIA.module.edge import Edge
from DIA.utils.env_utils import create_env_plan

import os

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

def load_edge_model(edge_model_path, env, args):
    if edge_model_path is not None:
        edge_model_dir = osp.dirname(edge_model_path)
        edge_model_vv = json.load(open(osp.join(edge_model_dir, 'best_state.json')))
        edge_model_vv['eval'] = 1
        edge_model_vv['n_epoch'] = 1
        edge_model_vv['edge_model_path'] = edge_model_path
        edge_model_vv['env'] = args.env
        edge_model_args = vv_to_args(edge_model_vv)

        edge = Edge(edge_model_args, env=env)
        print('edge GNN model successfully loaded from ', edge_model_path, flush=True)
    else:
        print("no edge GNN model is loaded")
        edge = None

    return edge


def load_dynamics_model(args, env, edge):
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
    model_vv['dataset']['pred_time_interval'] = args.dataset.pred_time_interval
    model_vv['cuda_idx'] = args.cuda_idx
    model_vv['partial_dyn_path'] = args.partial_dyn_path
    model_vv['env']['env_shape'] = args.env.env_shape
    model_vv['dataset']['env_shape'] = args.env.env_shape
    args = vv_to_args(model_vv)

    dynamics = DynamicIA(args, edge=edge, env=env)
    return dynamics

def plan(args, log_dir):

    mp.set_start_method('forkserver', force=True)

    env, render_env = create_env_plan(args.env)

    edge = load_edge_model(args.edge_model_path, env, args)
    dyn_model = load_dynamics_model(args, env, edge)

    # compute camera matrix
    camera_pos, camera_angle = env.get_camera_params()
    matrix_world_to_camera = get_matrix_world_to_camera(cam_angle=camera_angle, cam_pos=camera_pos)

    # build random shooting planner
    planner = MPCPlanner(
        dynamics=dyn_model,
        reward_model=cloth_drop_reward_fuc,
        matrix_world_to_camera=matrix_world_to_camera,
        env=env,
        args=args,
    )

    initial_states, configs, action_trajs, all_infos, all_normalized_performance, = [], [], [], [], []

    for episode_idx in range(args.configurations):
        # setup environment, ensure the same initial configuration
        env.reset(config_id=episode_idx)
        env.action_tool.update_picker_boundary([-0.3, 0.0, -0.5], [0.5, 2, 0.5])

        config = env.get_current_config()
        cloth_xdim, cloth_ydim = config['ClothSize']

        config_id = env.current_config_id
        scene_params = [env.cloth_particle_radius, cloth_xdim, cloth_ydim, config_id]


        downsample_idx, downsample_x_dim, downsample_y_dim = downsample(cloth_xdim, cloth_ydim, args.dataset.down_sample_scale)
        scene_params[1] = downsample_x_dim
        scene_params[2] = downsample_y_dim

        initial_state = env.get_state()
        initial_states.append(initial_state), configs.append(config)

        rewards, action_traj, infos, frames, frames_top = [], [], [], [], []
        gt_positions, gt_shape_positions, model_pred_particle_poses, model_pred_shape_poses, predicted_edges_all, predicted_performances, predicted_all_results = [], [], [], [], [], [], []
        actual_control_num = 0

        flex_states, start_poses, after_poses = [env.get_state()], [], []
        obses = [env.get_image(env.camera_width, env.camera_height)]

        for control_sequence_idx in range(args.sequence_steps):
            # prepare input data for planning
            cloth_mask, rgb, depth = get_rgbd_and_mask(env, args.sensor_noise)
            world_coordinates = get_world_coords(rgb, depth, env)[:, :, :3].reshape((-1, 3))
            pointcloud = world_coordinates[depth.flatten() > 0].astype(np.float32)
            # stop if the cloth is dragged out-of-view
            if len(pointcloud) == 0:
                print("cloth dragged out of camera view!")
                break
            voxel_pc = voxelize_pointcloud(pointcloud, args.dataset.voxel_size)
            _, observable_particle_indices = get_observable_particle_index_3(voxel_pc, env.get_state()['particle_pos'].reshape(-1,4)[downsample_idx,:3])

            vel_history = np.zeros((len(observable_particle_indices), args.dataset.n_his * 3), dtype=np.float32)

            # update velocity history
            if len(gt_positions)>1:
                for i in range(min(len(gt_positions)-1,args.dataset.n_his-1)):
                    start_index = (min(control_sequence_idx, args.dataset.n_his)) * (-3) + i * 3
                    vel_history[:, start_index:start_index+3] = (gt_positions[i+1][0][observable_particle_indices] - gt_positions[i][0][observable_particle_indices]) / (args.dataset.dt * args.dataset.pred_time_interval)

                vel_history[:, -3:] = (voxel_pc -gt_positions[-1][0][observable_particle_indices]) / (args.dataset.dt * args.dataset.pred_time_interval*0.8)

            elif len(gt_positions) == 1:
                vel_history[:, -3:] = (voxel_pc -gt_positions[-1][0][observable_particle_indices]) / (args.dataset.dt * args.dataset.pred_time_interval*0.8)
            else:
                pass # vel_history is already zeros

            picker_position, picked_points = env.action_tool._get_pos()[0], [-1, -1]
            data = {
                'pointcloud': voxel_pc,
                'vel_his': vel_history,
                'picker_position': picker_position,
                'action': env.action_space.sample(),  # action will be replaced by sampled action later
                'picked_points': picked_points,
                'scene_params': scene_params,
                'partial_pc_mapped_idx': observable_particle_indices,
                'downsample_idx': downsample_idx,
                'target_pos': config['target_pos'],
            }
            if args.env.env_shape != None:
                data['shape_size'] = config['shape_size']
                data['shape_pos'] = config['shape_pos']
                data['shape_quat'] = config['shape_quat']

            if control_sequence_idx==0:
                init_results, step_mid_sampled = planner.init_traj(data)
                # visualize all middle state
                middle_state_pc = [init_results[i]['model_positions'][step_mid_sampled[i]] for i in range(len(init_results))]
                middle_state_shape = [init_results[i]['shape_positions'][step_mid_sampled[i]] for i in range(len(init_results))]

            action_sequence, model_pred_particle_pos, model_pred_shape_pos, cem_info, predicted_edges, results \
                = planner.get_action(data, control_sequence_idx=control_sequence_idx)

            print("config {} control sequence idx {}".format(config_id, control_sequence_idx), flush=True)

            # record model predicted data for plotting
            model_pred_particle_poses.append(model_pred_particle_pos)
            model_pred_shape_poses.append(model_pred_shape_pos)
            predicted_edges_all.append(predicted_edges)
            predicted_performances.append(cem_info['highest_return'])
            predicted_all_results.append(results)

            # decompose the large 5-step action to be small 1-step actions to execute

            if args.dataset.pred_time_interval >= 2:
                action_sequence_small = np.zeros((args.dataset.pred_time_interval, 8))
                action_sequence_small[:, :] = (action_sequence[0,:]) / args.dataset.pred_time_interval

                action_sequence_small[action_sequence_small[:, 3]>0, 3] =1
                action_sequence_small[action_sequence_small[:, 7]>0, 7] =1
                action_sequence_executed = action_sequence_small

            # execute the planned action
            gt_positions.append(np.zeros((len(action_sequence_executed), len(downsample_idx), 3)))
            gt_shape_positions.append(np.zeros((len(action_sequence_executed), 2, 3)))
            for t_idx, ac in enumerate(action_sequence_executed):
                _, reward, done, info = env.step(ac, record_continuous_video=True, img_size=360)

                frames.extend(info['flex_env_recorded_frames'])
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

        #####################
        # SAVE EVERYTHING
        #####################

        log_dir_episode = osp.join(log_dir, str(episode_idx))
        os.makedirs(log_dir_episode, exist_ok=True)

        #draw target pos

        # draw middle state
        for i in range(len(middle_state_pc)):
            draw_middle_state([middle_state_pc[i]], [middle_state_shape[i]], render_env, config_id,
                       matrix_world_to_camera, log_dir_episode, i, env)

        draw_target_gif(env, matrix_world_to_camera, log_dir_episode, args.env.env_name, episode_idx)

        # # draw the model predictions for each step
        # draw_prediction_step(actual_control_num, gt_positions, gt_shape_positions, render_env,
        #                  model_pred_particle_poses, model_pred_shape_poses, config_id, downsample_idx,
        #                     predicted_edges_all, matrix_world_to_camera, log_dir_episode, episode_idx,
        #                  args, predicted_all_results, env)

        # draw the groundtruth trajectory
        draw_gt_trajectory(gt_positions, gt_shape_positions, render_env, config_id, downsample_idx,
                       matrix_world_to_camera, log_dir_episode, episode_idx, env)

        # plot the performance curve
        transformed_info = transform_info([infos])
        plot_performance_curve(transformed_info, log_dir_episode, episode_idx, predicted_performances, gt_shape_positions)

        # make gif from two views
        make_result_gif(frames, env, matrix_world_to_camera, episode_idx, logger, args, frames_top)


        normalized_performance_traj = [info['normalized_performance'] for info in infos]
        all_normalized_performance.append(normalized_performance_traj)
        action_trajs.append(action_traj)
        all_infos.append(infos)

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

    # end
    print('planning finished')
