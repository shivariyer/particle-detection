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
	def __init__(self, list_IDs, files, ppm, tranform):
		self.filenames = files
		self.list_IDs = list_IDs
		self.ppm = ppm
		self.transform = tranform

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		ID = self.list_IDs[index]
		cv_in = cv2.imread('../img/'+self.filenames[ID]+'.jpg',1) # color
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
	data = pd.read_csv('../final_data.csv')
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
	model = LeUNet()
	model = torch.nn.DataParallel(model).cuda()
	model.load_state_dict(torch.load("model_hazy_best.pth"),strict=False) # on GPU
	criterion = nn.MSELoss().cuda()
	optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
	train_dataset = Dataset(ids_train, files_train, ppm_train, transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize(mean=[0.5231, 0.5180, 0.5115],std=[0.2014, 0.2018, 0.2100]),])) # normalize
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
	val_dataset = Dataset(ids_val, files_val, ppm_val, transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize(mean=[0.5231, 0.5180, 0.5115],std=[0.2014, 0.2018, 0.2100]),]))
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

	best_loss = 1e5
	for epoch in range(500):
		train_loss = train(train_loader,model,criterion,optimizer)
		val_loss = val(val_loader,model,criterion)
		print('Epoch: %d, MSE train set: %.8f' % (epoch+1, train_loss))
		print('Epoch: %d, MSE val set: %.8f\n' % (epoch+1, val_loss))
		if val_loss < best_loss:
			torch.save(model.state_dict(),'model_pm_best.pth')
			best_loss = val_loss

if __name__ == "__main__":
	main()


