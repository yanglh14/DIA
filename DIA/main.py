import os.path as osp
import json
import hydra
import os
from omegaconf import DictConfig, OmegaConf

from chester import logger

from DIA.utils.utils import set_resource, configure_logger, configure_seed
from DIA.task.train_dy import train
from DIA.task.gen_data import gen_data

@hydra.main(config_path="cfg", config_name="cfg")
def main(args: DictConfig) -> None:

    # Get the directory of the current script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Change the working directory to the script directory
    os.chdir(script_dir)

    set_resource()  # To avoid pin_memory issue
    configure_logger(args.log_dir, args.exp_name)
    configure_seed(args.seed)

    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(OmegaConf.to_container(args, resolve=True), f, indent=2, sort_keys=True)
    assert args.task in ['gen_data', 'train_dy', 'train_edge', 'plan'], 'Invalid task'
    if args.task == 'train_dy':
        train(args.train)
    elif args.task == 'gen_data':
        gen_data(args.gen_data)
    else:
        pass
if __name__ == "__main__":
    main()
