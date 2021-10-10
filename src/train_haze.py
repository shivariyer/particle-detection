#import argparse
#import os
#import random
#import time
#import sys
import numpy as np
import pandas as pd
#import shutil

import torch
import torch.nn as nn
#import torch.nn.parallel
import torch.optim
#iimport torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
#import torchvision.datasets as datasets
#import torchvision.models as models
from PIL import Image
import cv2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, files, tranform, transform2):
        self.filenames = files
        self.list_IDs = list_IDs
        self.transform = tranform
        self.transform2 = transform2

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        cv_in = cv2.imread('/scratch/ab9738/pollution_img/data/'+self.filenames[ID]+'.jpg',1) # color
        cv_out = cv2.imread('/scratch/ab9738/pollution_img/data/'+self.filenames[ID]+'_trans.jpg',0) # grayscale
        #if cv_in is None:
            #print('input: '+self.filenames[ID])
        #if cv_out is None:
            #print('output: '+self.filenames[ID])
        pil_in = Image.fromarray(cv_in)
        pil_out = Image.fromarray(cv_out)
        X = self.transform(pil_in)
        Y = self.transform2(pil_out)
        return X,Y

def double_conv(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),nn.ReLU(inplace=True),nn.Conv2d(out_channels, out_channels, 3, padding=1),nn.ReLU(inplace=True))

class UNet(nn.Module):

    def __init__(self):
        super(UNet,self).__init__()
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') #, align_corners=True)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out

def train(train_loader,model,criterion,optimizer,epoch):
    model.train()
    total_loss = 0.0
    epoch_samples = 0
    for x, y in train_loader:
        epoch_samples += x.size(0)
        #y = y.float()
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        x_var = torch.autograd.Variable(x)
        y_var = torch.autograd.Variable(y)

        yhat = model(x_var)
        loss = criterion(yhat,y_var)
        total_loss += loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (total_loss/epoch_samples)


def main():
    data = pd.read_csv('../china_data.csv')
    data_train = data.sample(frac=1) # unordered rn, data (if want ordered)
    files_train = list(data_train['filename'])
    ids_train = [i for i in range(len(files_train))]
    data = None
    model = UNet()
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
    train_dataset = Dataset(ids_train, files_train, transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize(mean=[0.5231, 0.5180, 0.5115],std=[0.2014, 0.2018, 0.2100]),]), transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),])) # normalize
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)

    best_loss = 1e5
    for epoch in range(20):
        loss = train(train_loader,model,criterion,optimizer,epoch)
        print('Epoch: %d, MSE: %.8f' % (epoch+1, loss))
        if loss < best_loss:
            torch.save(model.state_dict(),r'model_haze_china.pth')
            best_loss = loss

if __name__ == "__main__":
    main()


