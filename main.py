import torch
from torch import nn
from torch import optim
from dataset.MNIST import MNIST
from models.model import Network
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from trainer.trainer import Trainer
from utils.utils import read_cfg, get_optimizer

cfg = read_cfg('config/mnist_adam_lr1e-3.yaml')

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30, fill=(1,)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define dataset
trainset = MNIST(csv_file='./data/train.csv', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

# Define loss and optimizer
net = Network()
criterion = nn.CrossEntropyLoss()

optimizer = get_optimizer(cfg, net)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.8)

trainer = Trainer(cfg, net, optimizer, criterion, trainloader, trainloader, scheduler)
trainer.train()


