import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class MNIST(Dataset):
    def __init__(self, csv_file, transform=None):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx, 1:].values.astype(np.uint8).reshape(28, 28)
        label = self.data.iloc[idx, 0]

        # Convert np array to PIL Image object, transforms work on PIL Image object
        image = Image.fromarray(image, mode='L')
        
        if self.transform:
            image = self.transform(image)
        return image, label
