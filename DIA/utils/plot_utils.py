import numpy as np
import pyflex
import os
import os.path as osp
import copy

from softgym.utils.visualization import save_numpy_as_gif
import matplotlib.pyplot as plt
from DIA.utils.camera_utils import get_matrix_world_to_camera, get_world_coords, get_observable_particle_index_3

from DIA.utils.utils import (
    downsample, transform_info, draw_planned_actions, visualize, draw_edge,
    cloth_drop_reward_fuc, voxelize_pointcloud, vv_to_args, cem_make_gif, configure_seed, configure_logger, draw_target_pos
)

def seg_3d_figure(data: np.ndarray, labels: np.ndarray, labelmap=None, sizes=None, fig=None):
    import plotly.colors as pc
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly.figure_factory as ff

    # Create a figure.
    if fig is None:
        fig = go.Figure()

    # Find the ranges for visualizing.
    mean = data.mean(axis=0)
    max_x = np.abs(data[:, 0] - mean[0]).max()
    max_y = np.abs(data[:, 1] - mean[1]).max()
    max_z = np.abs(data[:, 2] - mean[2]).max()
    all_max = max(max(max_x, max_y), max_z)

    # Colormap.
    cols = np.array(pc.qualitative.Alphabet)
    labels = labels.astype(int)
    for label in np.unique(labels):
        subset = data[np.where(labels == label)]
        subset = np.squeeze(subset)
        if sizes is None:
            subset_sizes = 1.5
        else:
            subset_sizes = sizes[np.where(labels == label)]
        color = cols[label % len(cols)]
        if labelmap is not None:
            legend = labelmap[label]
        else:
            legend = str(label)
        fig.add_trace(
            go.Scatter3d(
                mode="markers",
                marker={"size": subset_sizes, "color": color, "line": {"width": 0}},
                x=subset[:, 0],
                y=subset[:, 1],
                z=subset[:, 2],
                name=legend,
            )
        )
    fig.update_layout(showlegend=True)

    # This sets the figure to be a cube centered at the center of the pointcloud, such that it fits
    # all the points.
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[mean[0] - all_max, mean[0] + all_max]),
            yaxis=dict(nticks=10, range=[mean[1] - all_max, mean[1] + all_max]),
            zaxis=dict(nticks=10, range=[mean[2] - all_max, mean[2] + all_max]),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=1.0, y=0.75),
    )
    return fig

def draw_target_gif(env, matrix_world_to_camera, log_dir_episode, env_name, episode_idx):
    target_pos = env.get_current_config()['target_pos']
    curr_pos = pyflex.get_positions().reshape((-1, 4))
    curr_pos[:, :3] = target_pos
    pyflex.set_positions(curr_pos)
    _img = env.get_image(env.camera_width, env.camera_height)
    _imgs = [_img, _img]
    for i in range(len(_imgs)):
        _imgs[i] = draw_target_pos(_imgs[i], env.get_current_config()['target_pos'], matrix_world_to_camera[:3, :],
                                   env.camera_height, env.camera_width, env._get_key_point_idx())
    cem_make_gif([_imgs], log_dir_episode, env_name + '{}_target.gif'.format(episode_idx))

def draw_prediction_step(actual_control_num, gt_positions, gt_shape_positions,render_env,
                         model_pred_particle_poses,model_pred_shape_poses, config_id, downsample_idx,
                            predicted_edges_all, matrix_world_to_camera, log_dir_episode, episode_idx,
                         args, predicted_all_results, env):
    # make gif visuals of the model predictions and groundtruth rollouts
    for control_sequence_idx in range(0, actual_control_num):
        # subsample the actual rollout (since we decomposed the 5-step action to be 1-step actions)
        subsampled_gt_pos, subsampled_shape_pos = [], []
        for t in range(5):
            subsampled_gt_pos.append(gt_positions[control_sequence_idx][t])
            subsampled_shape_pos.append(gt_shape_positions[control_sequence_idx][t])

        frames_model = visualize(render_env, model_pred_particle_poses[control_sequence_idx],
                                 model_pred_shape_poses[control_sequence_idx],
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
                _frames[t] = draw_target_pos(_frames[t], env.get_current_config()['target_pos'],
                                             matrix_world_to_camera[:3, :],
                                             env.camera_height, env.camera_width, env._get_key_point_idx())

            save_numpy_as_gif(_frames,
                              osp.join(log_dir_episode, '{}-{}-{}.gif'.format(episode_idx, control_sequence_idx, name)))

        if control_sequence_idx == 0:
            model_pred_particle_poses_all = []
            for i in range(args.shooting_number):

                frames_model_sample_action = visualize(render_env,
                                                       predicted_all_results[control_sequence_idx][i]['model_positions'],
                                                       model_pred_shape_poses[control_sequence_idx],
                                                       config_id,
                                                       range(model_pred_particle_poses[control_sequence_idx].shape[1]))

                _frames = np.array(frames_model_sample_action)
                for t in range(len(_frames)):
                    _frames[t] = draw_target_pos(_frames[t], env.get_current_config()['target_pos'],
                                                 matrix_world_to_camera[:3, :],
                                                 env.camera_height, env.camera_width, env._get_key_point_idx())

                log_dir_episode_sample_action = osp.join(log_dir_episode,
                                                         'sample_action_{}'.format(control_sequence_idx))
                os.makedirs(log_dir_episode_sample_action, exist_ok=True)
                save_numpy_as_gif(_frames, osp.join(log_dir_episode_sample_action,
                                                    '{}-{}-ActionSampling{}.gif'.format(episode_idx,
                                                                                        control_sequence_idx, i)))
def draw_gt_trajectory(gt_positions, gt_shape_positions, render_env, config_id, downsample_idx,
                       matrix_world_to_camera, log_dir_episode, episode_idx, env):
    _gt_positions = np.array(gt_positions).reshape(-1, len(downsample_idx), 3)
    _gt_shape_positions = np.array(gt_shape_positions).reshape(-1, 2, 3)
    frames_gt = visualize(render_env, _gt_positions, _gt_shape_positions, config_id, downsample_idx)
    for t in range(len(frames_gt)):
        frames_gt[t] = draw_target_pos(frames_gt[t], env.get_current_config()['target_pos'],
                                       matrix_world_to_camera[:3, :],
                                       env.camera_height, env.camera_width, env._get_key_point_idx())
    save_numpy_as_gif(np.array(frames_gt), osp.join(log_dir_episode, '{}-gt.gif'.format(episode_idx)))

def draw_middle_state(gt_positions, gt_shape_positions, render_env, config_id,
                       matrix_world_to_camera, log_dir_episode,idx, env):
    _gt_positions = np.array(gt_positions)
    _gt_shape_positions = np.array(gt_shape_positions)
    frames_gt = visualize(render_env, _gt_positions, _gt_shape_positions, config_id)
    for t in range(len(frames_gt)):
        frames_gt[t] = draw_target_pos(frames_gt[t], env.get_current_config()['target_pos'],
                                       matrix_world_to_camera[:3, :],
                                       env.camera_height, env.camera_width, env._get_key_point_idx())
    save_numpy_as_gif(np.array(frames_gt), osp.join(log_dir_episode, '{}.gif'.format(idx)))


def plot_performance_curve(transformed_info, log_dir_episode, episode_idx, predicted_performances, gt_shape_positions):
    performance_gt = transformed_info['performance'][0]
    performance_pre = np.array(predicted_performances)
    plt.plot(performance_gt, label='gt')
    plt.plot(performance_pre, label='pre')
    plt.legend()
    plt.savefig(osp.join(log_dir_episode, '{}-performance.png'.format(episode_idx)))
    plt.close()

    _gt_shape_positions = np.array(gt_shape_positions).reshape(-1, 2, 3)
    plt.plot(_gt_shape_positions[:, 0, 0], _gt_shape_positions[:, 0, 1])
    plt.savefig(os.path.join(log_dir_episode, 'X-Z.png'))
    plt.close()

def make_result_gif(frames, env, matrix_world_to_camera, episode_idx, logger, args,frames_top):
    for i in range(len(frames)):
        frames[i] = draw_target_pos(frames[i], env.get_current_config()['target_pos'], matrix_world_to_camera[:3, :],
                                    env.camera_height, env.camera_width, env._get_key_point_idx())

    cem_make_gif([frames], logger.get_dir(), args.env.env_name + '{}.gif'.format(episode_idx))


    matrix_world_to_camera_ = get_matrix_world_to_camera(cam_angle=np.array([1.57, -1.57, 0]),
                                                         cam_pos=np.array([0.2, 1.0, 0]))
    for i in range(len(frames_top)):
        frames_top[i] = draw_target_pos(frames_top[i], env.get_current_config()['target_pos'],
                                        matrix_world_to_camera_[:3, :],
                                        env.camera_height, env.camera_width, env._get_key_point_idx())
    cem_make_gif([frames_top], logger.get_dir(), args.env.env_name + '{}_top.gif'.format(episode_idx))
