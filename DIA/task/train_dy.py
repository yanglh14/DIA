from DIA.utils.utils import vv_to_args
from DIA.module.dynamics import DynamicIA
from DIA.module.edge import Edge
from DIA.utils.env_utils import create_env

import json
import os.path as osp

def train(args):

    if args.local:
        env = create_env(args.env)
    else:
        env = None

    # load vcd_edge
    if args.edge_model_path is not None:
        edge_model_vv = json.load(open(osp.join(args.edge_model_path, 'variant.json')))
        edge_model_args = vv_to_args(edge_model_vv)
        dia_edge = Edge(edge_model_args, env=env)
        dia_edge.load_model(args.edge_model_path)
        print('EdgeGNN successfully loaded from ', args.edge_model_path, flush=True)
    else:
        dia_edge = None

    dynamic_model = DynamicIA(args, env, dia_edge)

    dynamic_model.train()