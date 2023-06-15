import os
import os.path as osp
import copy
import cv2
import json
import wandb
import numpy as np
import scipy
from tqdm import tqdm
from chester import logger

import torch
import torch_geometric

from softgym.utils.visualization import save_numpy_as_gif

from DIA.models import GNN
from DIA.dataset import ClothDataset
from DIA.utils.data_utils import AggDict
from DIA.utils.utils import extract_numbers, pc_reward_model, visualize
from DIA.utils.camera_utils import get_matrix_world_to_camera, project_to_image

class DynamicIA(object):

    def __init__(self, args, env):

        self.env = env
        self.args = args
        self.device = torch.device(self.args.cuda_idx)

        self.train_mode = args.train_mode
        self.input_types = ['full', 'vsbl'] if self.train_mode == 'graph_imit' else [self.train_mode]
        self.models, self.optims, self.schedulers = {}, {}, {}

        # create model
        for m in self.input_types:

            self.models[m] = GNN(args, decoder_output_dim=3, name=m,
                                 use_reward=False if self.train_mode == 'vsbl' else True)  # Predict acceleration
            lr = getattr(self.args, m + '_lr') if hasattr(self.args, m + '_lr') else self.args.lr
            self.optims[m] = torch.optim.Adam(self.models[m].param(), lr=lr, betas=(self.args.beta1, 0.999))
            self.schedulers[m] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optims[m], 'min', factor=0.8,
                                                                            patience=3, verbose=True)
            self.models[m].to(self.device)

        print("DIA dynamics models created")

        #Todo load model if needed

        # create dataset
        self.datasets = {phase: ClothDataset(args, self.input_types, phase, env) for phase in ['train', 'valid']}

        follow_batch = ['x_{}'.format(t) for t in self.input_types]
        self.dataloaders = {x: torch_geometric.data.DataLoader(
            self.datasets[x], batch_size=args.batch_size, follow_batch=follow_batch,
            shuffle=True if x == 'train' else False, drop_last=True,
            num_workers=args.num_workers, pin_memory=True, prefetch_factor=5 if args.num_workers > 0 else 2)
            for x in ['train', 'valid']}

        print("DIA datasets created")

        self.mse_loss = torch.nn.MSELoss()
        self.log_dir = logger.get_dir()

    def generate_dataset(self):
        os.system('mkdir -p ' + self.args.dataf)
        for phase in ['train', 'valid']:
            self.datasets[phase].generate_dataset()
        print('Dataset generated in', self.args.dataf)