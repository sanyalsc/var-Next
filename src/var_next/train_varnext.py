import os
import argparse

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import json

from var_next.vnext import varNext


def set_up_dataset(data_dir):
    dataset = ImageFolder(data_dir,transform=transforms.ToTensor())
    train, val = random_split(dataset,[0.8, 0.2])
    train_set = DataLoader(train, batch_size=64, shuffle=True)
    val_set = DataLoader(val, batch_size=64, shuffle=True)
    return train_set, val_set


def train(cfg_file,data_dir, n_epoch=5, result_dir='/scratch/ejg8qa/360_results'):
    test_id = os.path.splitext(os.path.basename(cfg_file))[0]
    output_dir = os.path.join(result_dir,test_id)
    os.makedirs(output_dir,exist_ok=True)
    
    cfg = json.load(cfg_file)
    train_loader, val_loader = set_up_dataset(data_dir)

    model = varNext(cfg)
    optim = torch.optim.Adam(model.parameters(), lr=cfg['learn_rate'], weight_decay=0.01)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    model.to(device)

    with open(os.path.join(output_dir,'logfile.txt'),'w') as rfi:
        for epoch in range(n_epoch):
            kl = False
            if epoch >1:
                kl = True
            train_loss = train_epoch(model,device,train_loader,optim, kl,rfi)
            val_loss = test_epoch(model,device,val_loader)
            rfi.write(f'\n EPOCH {epoch+1}/{n_epoch}: train loss {train_loss}, val loss {val_loss}')
            torch.save(model.state_dict(), os.path.join(output_dir,'model_wts.pt'))

    torch.save(model.state_dict(), os.path.join(output_dir,'model_wts.pt'))


def train_epoch(vae, device, dataloader, optimizer,kl=False, rfi=None):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, _ in dataloader: 
        # Move tensor to the proper device
        x = x.to(device)
        y = vae(x)
        # Evaluate loss
        loss = torch.nn.functional.mse_loss(y,x,reduction='sum')
        if kl:
            loss = loss + vae.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        if rfi:
            rfi.write(f'\n partial train loss (single batch): {loss.item()}')

        print(f'partial train loss (single batch): {loss.item()}\r')
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
            y = vae(x)
            loss = torch.nn.functional.mse_loss(y,x,reduction='sum') + vae.kl
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net-cfg',required=True,help='Network configuration')
    parser.add_argument('--data',required=True,help='Input directory of images')
    parser.add_argument('--output',default='/scratch/ejg8qa/360_results',help='output directory for results and weights')
    parser.add_argument('--n-epoch',default=5,help='number of epochs')
    return parser.parse_args()

if __name__ == '__main__':
    args = load_args()
    train(args.net_cfg,args.data,args.output)