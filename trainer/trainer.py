import os
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from trainer.base import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, criterion, trainloader, testloader, lr_scheduler):
        super().__init__(cfg, model, optimizer, criterion)
        self.trainloader = trainloader
        self.testloader = testloader
        self.lr_scheduler = lr_scheduler


    def load_model(self):
        raise NotImplementedError


    def save_model(self, epoch):
        save_name = '{}-{}.pth'.format(self.cfg['model']['name'], epoch)
        save_path = os.path.join(self.cfg['output_dir'], save_name)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        }


    def train_one_epoch(self, epoch):
        self.model.train()

        for i, (data, target) in enumerate(self.trainloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar('training loss', loss.item(), epoch * len(self.trainloader) + i)

    
    def train(self):
        for epoch in range(self.cfg['train']['num_epochs']):
            self.train_one_epoch(epoch)
            self.validate(epoch)


    def validate(self, epoch):
        self.model.eval()
        accuracy = 0.0
        with torch.no_grad():
            for i, (data, target) in enumerate(self.testloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                values, indices = torch.max(output, 1)
                equal = target.eq(indices)
                accuracy += torch.mean(equal.type(torch.FloatTensor)).item()
        self.writer.add_scalar('validation acc', accuracy/len(self.testloader), epoch)
        return accuracy
        
