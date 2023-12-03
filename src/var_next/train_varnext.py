import os
import argparse
import time

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import json

from var_next.vnext import varNext
from annotation_dataloader import varDataset

def set_up_dataset(data_dir,annotation_path,gray=False):
    train_path = os.path.join(data_dir,'train')
    val_path = os.path.join(data_dir,'val')
    train = varDataset(train_path,annotation_path,gray=gray)
    val = varDataset(val_path,annotation_path,gray=gray)
    train_set = DataLoader(train, batch_size=64, shuffle=True)
    val_set = DataLoader(val, batch_size=64, shuffle=True)
    return train_set, val_set


def train(cfg_file,data_dir, n_epoch=5, result_dir='/scratch/ejg8qa/360_results', annotation_path='/scratch/ejg8qa/log_images_320/master_annot.csv'):
    
    with open(cfg_file,'r') as cfi:
        cfg = json.load(cfi)
    test_id = os.path.splitext(os.path.basename(cfg_file))[0]
    output_dir = os.path.join(result_dir,f'{test_id}_beta_{cfg["beta"]}')
    os.makedirs(output_dir,exist_ok=True)
    if "gray" in cfg:
        gray = cfg['gray']
    else:
        gray = False
    train_loader, val_loader = set_up_dataset(data_dir,annotation_path,gray=gray)
    print(f'running with :{cfg}')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    model = varNext(cfg, device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg['learn_rate'], weight_decay=0.01)

    model.to(device)
    last_t = time.time()
    best_val = torch.inf
    kl = True
    beta = cfg['beta']
    with open(os.path.join(output_dir,'logfile.txt'),'w') as rfi:
        for epoch, loss in enumerate(run_epoch(model,device,train_loader,val_loader,optim,kl,rfi,beta)):
            train_loss, val_loss = loss
            rfi.write(f'\n EPOCH {epoch+1}/{n_epoch} took {time.time()-last_t}s: train loss {train_loss}, val loss {val_loss}')
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), os.path.join(output_dir,'model_wts.pt'))
            last_t=time.time()

            if epoch==n_epoch:
                break


def run_epoch(model, device, train_loader, val_loader, optim ,kl=False, rfi=None, beta=2):
    while True:
        train_loss = train_epoch(model,device,train_loader,optim, kl,rfi,beta)
        val_loss = test_epoch(model,device,val_loader,beta)
        yield train_loss, val_loss


def train_epoch(vae, device, dataloader, optimizer,kl=False, rfi=None, beta=2):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x, mask in dataloader: 
        # Move tensor to the proper device
        x = x.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        y = vae(x)
        xmask = x * mask
        ymask = y * mask
        shape = x.shape
        scalef = torch.sum(mask,dim=(1,2,3))/(shape[1]*shape[2]*shape[3])
        scalef[scalef!=0] = 1/scalef[scalef!=0]
        # Evaluate loss
        l1 = torch.nn.functional.mse_loss(y,x,reduction='sum')
        l2 = torch.sum(scalef * torch.sum(torch.nn.functional.mse_loss(ymask,xmask,reduction='none'),dim=(1,2,3)))
        
        loss = l1 + l2
        if kl:
            loss = loss + beta*vae.kl

        # Backward pass
        loss.backward()
        optimizer.step()
        # Print batch loss
        if rfi:
            rfi.write(f'\n sb loss, l1:{l1}, l2: {l2}, B*kl: {beta*vae.kl}')

        print(f'partial train loss (single batch): {loss.item()}\r')
        train_loss+=loss.item()

    return train_loss / len(dataloader.dataset)


def test_epoch(vae, device, dataloader, beta=2):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    for x, mask in dataloader:
        # Move tensor to the proper device
        x = x.to(device)
        mask = mask.to(device)
        # Decode data
        y = vae(x)
        xmask = x * mask
        ymask = y * mask
        shape = x.shape
        scalef = torch.sum(mask,dim=(1,2,3))/(shape[1]*shape[2]*shape[3])
        # Evaluate loss

        l1 = torch.nn.functional.mse_loss(y,x,reduction='sum')
        l2 = torch.sum(scalef * torch.sum(torch.nn.functional.mse_loss(ymask,xmask,reduction='none'),dim=(1,2,3)))
        
        loss = l1 + l2 + beta*vae.kl
        val_loss += loss.item()

    return val_loss / len(dataloader.dataset)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net-cfg',required=True,help='Network configuration')
    parser.add_argument('--data',required=True,help='Input directory of images')
    parser.add_argument('--output',default='/scratch/ejg8qa/360_results',help='output directory for results and weights')
    parser.add_argument('--n-epoch',default=5,type=int,help='number of epochs')
    #parser.add_argument('--annot-p',default=1,type=int,help='kld beta value')
    return parser.parse_args()

if __name__ == '__main__':
    args = load_args()
    train(args.net_cfg,args.data,args.n_epoch,args.output)