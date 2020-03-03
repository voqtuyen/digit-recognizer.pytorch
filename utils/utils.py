import yaml
import torch
from torch import optim


def get_optimizer(cfg, model):
    optimizer = None
    if cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    return optimizer


def read_cfg(cfg_file):
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
        return cfg

    
def get_device(cfg):
    if cfg['train']['device'] == 'cpu':
        return torch.device('cpu')
    elif cfg['train']['device'] == 'gpu':
        return torch.device('cuda')
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')