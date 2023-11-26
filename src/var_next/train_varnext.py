import torch
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import json

from var_next.var_next import varNext


def set_up_dataset(data_dir):
    dataset = ImageFolder(data_dir,transform=transforms.ToTensor())
    train, val = random_split(dataset,[0.8, 0.2])
    train_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    val_set = torch.utils.data.DataLoader(val, batch_size=64, shuffle=True)
    return train_set, val_set


def train(cfg_file,data_dir):
    cfg = json.load(cfg_file)
    train_set, val_set = set_up_dataset(data_dir)

    model = varNext(cfg)
    optim = torch.optim.Adam(model.parameters(), lr=cfg['learn_rate'], weight_decay=1e-5)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    model.to(device)


def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, _ in dataloader: 
        # Move tensor to the proper device
        x = x.to(device)
        x_hat = vae(x)
        # Evaluate loss
        loss = ((x - x_hat)**2).sum() + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss+=loss.item()

    return train_loss / len(dataloader.dataset)


def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x, _ in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            # calc KL divergence
            _ = vae.encoder(x)
            # Decode data
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)