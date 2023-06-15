import argparse
import copy
import os.path as osp
import json
from chester import logger

from DIA.utils.utils import set_resource,configure_logger, configure_seed
from DIA.dynamics import DynamicIA

def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='dia', help='Experiment name')
    parser.add_argument('--log_dir', type=str, default='data/log', help='Logging directory')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    #ENV
    parser.add_argument('--env_name', type=str, default='ClothDrop', help='Environment name')
    parser.add_argument('--cached_states_path', type=str, default='dia.pkl', help='Path to cached states')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of variations in the dataset')
    parser.add_argument('--partial_observable', type=bool, default=True, help="Whether only the partial point cloud can be observed")
    parser.add_argument('--particle_radius', type=float, default=0.00625, help='Particle radius for the cloth')
    ## pyflex shape state
    parser.add_argument('--shape_state_dim', type=int, default=14, help="[xyz, xyz_last, quat(4), quat_last(4)]")

    #Dataset
    parser.add_argument('--n_rollout', type=int, default=2000, help='Number of training trajectories')
    parser.add_argument('--time_step', type=int, default=100, help='Time steps per trajectory')
    parser.add_argument('--dt', type=float, default=1. / 100.)
    parser.add_argument('--pred_time_interval', type=int, default=5, help='Interval of timesteps between each dynamics prediction (model dt)')
    parser.add_argument('--train_valid_ratio', type=float, default=0.9, help="Ratio between training and validation")
    parser.add_argument('--dataf', type=str, default='./data/dia/', help='Path to dataset')
    parser.add_argument('--gen_data', type=int, default=1, help='Whether to generate dataset')
    parser.add_argument('--gen_gif', type=bool, default=1, help='Whether to also save gif of each trajectory (for debugging)')

    # Model
    parser.add_argument('--global_size', type=int, default=128, help="Number of hidden nodes for global in GNN")
    parser.add_argument('--n_his', type=int, default=5, help="Number of history step input to the dynamics")
    parser.add_argument('--down_sample_scale', type=int, default=3, help="Downsample the simulated cloth by a scale of 3 on each dimension")
    parser.add_argument('--voxel_size', type=float, default=0.0216)
    parser.add_argument('--neighbor_radius', type=float, default=0.045, help="Radius for connecting nearby edges")
    parser.add_argument('--use_rest_distance', type=bool, default=True, help="Subtract the rest distance for the edge attribute of mesh edges")
    parser.add_argument('--use_mesh_edge', type=bool, default=True)
    parser.add_argument('--collect_data_delta_move_min', type=float, default=0.01)
    parser.add_argument('--collect_data_delta_move_max', type=float, default=0.02)
    parser.add_argument('--proc_layer', type=int, default=10, help="Number of processor layers in GNN")
    parser.add_argument('--state_dim', type=int, default=18,
                        help="Dim of node feature input. Computed based on n_his: 3 x 5 + 1 dist to ground + 2 one-hot encoding of picked particle")
    parser.add_argument('--relation_dim', type=int, default=7, help="""Dim of edge feature input: 
        3 for directional vector + 1 for directional vector magnitude + 2 for one-hot encoding of mesh or collision edge + 1 for rest distance
        """)

    #Training
    parser.add_argument('--train_mode', type=str, default='vsbl', help='Should be in ["vsbl", "graph_imit", "full"]')
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fixed_lr', type=bool, default=False, help='By default, decaying lr is used.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for dataloader')
    parser.add_argument('--eval', type=int, default=0, help='Whether to just evaluating the model')
    parser.add_argument('--nstep_eval_rollout', type=int, default=20, help='Number of rollout trajectory for evaluation')
    parser.add_argument('--save_model_interval', type=int, default=5, help='Save the model every N epochs during training')
    parser.add_argument('--use_wandb', type=bool, default=False, help='Use weight and bias for logging')


    args = parser.parse_args()

    return args

def create_env(args):
    from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
    assert args.env_name == 'ClothDrop'

    env_args = copy.deepcopy(env_arg_dict[args.env_name])  # Default args
    env_args['cached_states_path'] = args.cached_states_path
    env_args['num_variations'] = args.num_variations

    env_args['render'] = True
    env_args['headless'] = False
    env_args['render_mode'] = 'cloth' if args.gen_data else 'particle'
    env_args['camera_name'] = 'default_camera'
    env_args['camera_width'] = 360
    env_args['camera_height'] = 360

    env_args['num_picker'] = 2  # The extra picker is hidden and does not really matter
    env_args['picker_radius'] = 0.01
    env_args['picker_threshold'] = 0.00625
    env_args['action_repeat'] = 1

    if args.partial_observable and args.gen_data:
        env_args['observation_mode'] = 'cam_rgb'

    return SOFTGYM_ENVS[args.env_name](**env_args)
def main():
    set_resource()  # To avoid pin_memory issue
    args = get_default_args()
    env = create_env(args)

    configure_logger(args.log_dir, args.exp_name)
    configure_seed(args.seed)

    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    dynamic_model = DynamicIA(args, env)

    if args.gen_data:
        dynamic_model.generate_dataset()
    else:
        dynamic_model.train()
if __name__ == '__main__':
    main()