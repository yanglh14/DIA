from DIA.module.gen_dataset import DataCollector
from DIA.utils.env_utils import create_env

def gen_data(args):

    env = create_env(args.env)

    collectors = {phase: DataCollector(args.dataset, phase, env) for phase in ['train', 'valid']}

    # Todo: add test phase
    for phase in ['train']:
        collectors[phase].gen_dataset()

    print('Dataset generated in', args.dataset.dataf)