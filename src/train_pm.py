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
import torchvision.models as models
from PIL import Image
import cv2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, files, ppm, tranform):
        self.filenames = files
        self.list_IDs = list_IDs
        self.ppm = ppm
        self.transform = tranform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        cv_in = cv2.imread('/scratch/ab9738/pollution_img/data/'+self.filenames[ID]+'.jpg',1) # color
        pil_in = Image.fromarray(cv_in)
        X = self.transform(pil_in)
        y = self.ppm[ID]
        return X,y

def double_conv(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),nn.ReLU(inplace=True),nn.Conv2d(out_channels, out_channels, 3, padding=1),nn.ReLU(inplace=True))

class LeUNet(nn.Module):

    def __init__(self):
        super(LeUNet,self).__init__()
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
        self.features = nn.Sequential(nn.Conv2d(1, 6, 5),nn.ReLU(inplace=True),nn.MaxPool2d(2),nn.Conv2d(6, 16, 5),nn.ReLU(inplace=True),nn.MaxPool2d(2),)
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.estimator = nn.Sequential(nn.Linear(16*5*5,120),nn.ReLU(inplace=True),nn.Linear(120, 84),nn.ReLU(inplace=True),nn.Linear(84, 1),)

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
        x = self.conv_last(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),16*5*5)
        x = self.estimator(x)
        return x


class ResNetUNet(nn.Module):

    def __init__(self):
        super(LeUNet,self).__init__()
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
        self.features = models.resnet50()
        # self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.estimator = nn.Sequential(nn.Linear(1000,120),nn.ReLU(inplace=True),nn.Linear(120, 84),nn.ReLU(inplace=True),nn.Linear(84, 1),)

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
        x = self.conv_last(x)
        x = torch.cat((x,x,x),1)
        x = self.features(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0),16*5*5)
        x = self.estimator(x)
        return x


class StandardNet(nn.Module):

    def __init__(self, modelname='resnet101'):
        super(StandardNet, self).__init__()
        self.model = eval("models."+modelname+"()")
        self.estimator = nn.Sequential(nn.Linear(1000,120),nn.ReLU(inplace=True),nn.Linear(120,84),nn.ReLU(inplace=True),nn.Linear(84, 1),)

    def forward(self, x):
        x = self.model(x)
        x = self.estimator(x)
        return x

class EnsembleNet(nn.Module):
    def __init__(self):
        super(EnsembleNet, self).__init__()
        self.resnet50_pred = models.resnet50()
        self.vgg16_pred = models.vgg16()
        self.inceptionv3_pred = models.inception_v3(aux_logits=False)
        self.estimator = nn.Sequential(nn.Linear(3000,120),nn.ReLU(inplace=True),nn.Linear(120,84),nn.ReLU(inplace=True),nn.Linear(84,1),)

    def forward(self, x):
        a = self.resnet50_pred(x)
        b = self.vgg16_pred(x)
        c = self.inceptionv3_pred(x)
        x = torch.cat((a,b,c),1)
        x = self.estimator(x)
        return x

class NewReLU(nn.Module):
    """ Custom ReLU layer for baseline """
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha, self.beta = alpha, beta

    def forward(self, x):
        x = self.alpha*max(x,0) + min(0,x)*self.beta/x+1e-7
        return x

class EPAPLN(nn.Module):

    def __init__(self):
        super(EPAPLN, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3)
        self.conv2 = nn.Conv2d(32,32,3)
        self.maxpool = nn.MaxPool2d(3, 2)
        self.drop1 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(32,64,3)
        self.conv4 = nn.Conv2d(64,64,3)
        self.conv5 = nn.Conv2d(64,64,3)
        self.avgpool = nn.AvgPool2d(3, 2)
        self.drop2 = nn.Dropout(0.5)
        self.conv6 = nn.Conv2d(64,96,3)
        self.conv7 = nn.Conv2d(96,96,3)
        self.conv8 = nn.Conv2d(96,96,3)
        self.conv9 = nn.Conv2d(96,96,3)
        self.activation = nn.ReLU()
        self.adaptivepool = nn.AdaptiveAvgPool2d((2,2))
        self.estimator = nn.Sequential(nn.Linear(96*2*2,120),nn.ReLU(inplace=True),nn.Linear(120, 84),nn.ReLU(inplace=True),nn.Linear(84, 1),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.avgpool(x)
        x = self.drop2(x)
        x = self.conv6(x)
        x = self.activation(x)
        x = self.conv7(x)
        x = self.activation(x)
        x = self.conv8(x)
        x = self.activation(x)
        x = self.conv9(x)
        x = self.activation(x)
        x = self.adaptivepool(x)
        x = x.view(x.size(0),96*2*2)
        x = self.estimator(x)

        return x


def train(train_loader,model,criterion,optimizer):
    model.train()
    total_loss = 0.0
    epoch_samples = 0
    for x, y in train_loader:
        epoch_samples += x.size(0)
        y = y.float()
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        x_var = torch.autograd.Variable(x)
        y_var = torch.autograd.Variable(y)

        yhat = model(x_var)
        loss = criterion(yhat.squeeze(),y_var)
        total_loss += loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (total_loss/epoch_samples)


def val(val_loader,model,criterion):
    model.eval()
    total_loss = 0.0
    epoch_samples = 0
    #with torch.no_grad():
    for x, y in val_loader:
        epoch_samples += x.size(0)
        y = y.float()
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        x_var = torch.autograd.Variable(x)
        y_var = torch.autograd.Variable(y)

        yhat = model(x_var)
        loss = criterion(yhat.squeeze(),y_var)
        total_loss += loss.data.item()

    return (total_loss/epoch_samples)

def main():
    data = pd.read_csv('../all_data.csv')
    data_train = data.sample(frac=0.8,random_state=17)
    data_val = data.loc[~data.index.isin(data_train.index)]
    files_train = list(data_train['filename'])
    files_val = list(data_val['filename'])
    ppm_train = list(data_train['ppm'])
    ppm_val = list(data_val['ppm'])
    ids_train = [i for i in range(len(files_train))]
    ids_val = [i for i in range(len(files_val))]
    data = None
    data_train = None
    data_val = None
    # model = LeUNet()
    model = ResNetUNet()
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load("model_haze_all.pth"),strict=False) # on GPU
    # model = StandardNet('resnet50').cuda()
    # model = StandardNet('vgg16').cuda()
    # model = EPAPLN().cuda()
    # model = EnsembleNet().cuda()    
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    train_dataset = Dataset(ids_train, files_train, ppm_train, transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize(mean=[0.5231, 0.5180, 0.5115],std=[0.2014, 0.2018, 0.2100]),])) # normalize
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
    val_dataset = Dataset(ids_val, files_val, ppm_val, transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize(mean=[0.5231, 0.5180, 0.5115],std=[0.2014, 0.2018, 0.2100]),]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=12)

    best_loss = 1e5
    for epoch in range(500):
        train_loss = train(train_loader,model,criterion,optimizer)
        val_loss = val(val_loader,model,criterion)
        print('Epoch: %d, MSE train set: %.8f' % (epoch+1, train_loss))
        print('Epoch: %d, MSE val set: %.8f\n' % (epoch+1, val_loss))
        if val_loss < best_loss:
            torch.save(model.state_dict(),'resnetunet_pm_all.pth')
            best_loss = val_loss

if __name__ == "__main__":
    main()


