import os

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import pandas as pd


class varDataset(Dataset):
    def __init__(self,dirname,annot,gray=False):
        if gray:
            trans = transforms.Compose[
                transforms.ToTensor(),
                transforms.Grayscale()
            ]
        else:
            trans = transforms.ToTensor()

        self.data = ImageFolder(dirname,transform=trans)
        self.annot = pd.read_csv(annot)
        self.annot = self.annot.set_index('fname')
        
    def __getitem__(self, index):
        data = self.data[index]
        fname = os.path.basename(self.data.samples[index][0])
        idx = torch.tensor(self.annot.loc[fname].values,dtype=torch.int64)
        mask = torch.zeros_like(data[0],dtype=torch.float)
        mask[:,idx[1]:idx[3],idx[0]:idx[2]] = 1
        return data[0], mask

    def __len__(self):
        return len(self.data)