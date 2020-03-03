from torch.utils.tensorboard import SummaryWriter
from utils.utils import get_device

class BaseTrainer():
    def __init__(self, cfg, model, optimizer, criterion):
        self.cfg = cfg
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = SummaryWriter(log_dir=cfg['train']['log_dir'])
        self.device = get_device(cfg)
        self.model = model.to(self.device)


    def load_model(self):
        raise NotImplementedError


    def save_model(self):
        raise NotImplementedError


    def train_one_epoch(self):
        return NotImplementedError


    def train(self):
        return NotImplementedError


    def validate(self):
        return NotImplementedError
