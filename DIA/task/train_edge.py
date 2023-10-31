from DIA.utils.utils import vv_to_args
from DIA.module.dynamics import DynamicIA
from DIA.module.edge import Edge
from DIA.utils.env_utils import create_env

import json
import os.path as osp

def train_edge(args):

    if args.local:
        env = create_env(args.env)
    else:
        env = None

    edge = Edge(args, env)

    edge.train()