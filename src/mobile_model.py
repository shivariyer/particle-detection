#import argparse
#import os
#import random
#import time
#import sys
# import numpy as np
# import pandas as pd
#import shutil

import torch
import torch.nn as nn
#import torch.nn.parallel
import torch.optim
#iimport torch.multiprocessing as mp
import torch.utils.data
# import torchvision.transforms as transforms
#import torchvision.datasets as datasets
#import torchvision.models as models
# from PIL import Image
# import cv2
from torch.utils.mobile_optimizer import optimize_for_mobile
from train_pm import double_conv, LeUNet

# def double_conv(in_channels, out_channels):
# 	return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),nn.ReLU(inplace=True),nn.Conv2d(out_channels, out_channels, 3, padding=1),nn.ReLU(inplace=True))


# class LeUNet(nn.Module):

# 	def __init__(self):
# 		super(LeUNet,self).__init__()
# 		self.dconv_down1 = double_conv(3, 64)
# 		self.dconv_down2 = double_conv(64, 128)
# 		self.dconv_down3 = double_conv(128, 256)
# 		self.dconv_down4 = double_conv(256, 512)
# 		self.maxpool = nn.MaxPool2d(2)
# 		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') #, align_corners=True)
# 		self.dconv_up3 = double_conv(256 + 512, 256)
# 		self.dconv_up2 = double_conv(128 + 256, 128)
# 		self.dconv_up1 = double_conv(128 + 64, 64)
# 		self.conv_last = nn.Conv2d(64, 1, 1)
# 		self.features = nn.Sequential(nn.Conv2d(1, 6, 5),nn.ReLU(inplace=True),nn.MaxPool2d(2),nn.Conv2d(6, 16, 5),nn.ReLU(inplace=True),nn.MaxPool2d(2),)
# 		self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
# 		self.estimator = nn.Sequential(nn.Linear(16*5*5,120),nn.ReLU(inplace=True),nn.Linear(120, 84),nn.ReLU(inplace=True),nn.Linear(84, 1),)

# 	def forward(self, x):
# 		conv1 = self.dconv_down1(x)
# 		x = self.maxpool(conv1)
# 		conv2 = self.dconv_down2(x)
# 		x = self.maxpool(conv2)
# 		conv3 = self.dconv_down3(x)
# 		x = self.maxpool(conv3)
# 		x = self.dconv_down4(x)
# 		x = self.upsample(x)
# 		x = torch.cat([x, conv3], dim=1)
# 		x = self.dconv_up3(x)
# 		x = self.upsample(x)
# 		x = torch.cat([x, conv2], dim=1)
# 		x = self.dconv_up2(x)
# 		x = self.upsample(x)
# 		x = torch.cat([x, conv1], dim=1)
# 		x = self.dconv_up1(x)
# 		x = self.conv_last(x)
# 		x = self.features(x)
# 		x = self.avgpool(x)
# 		x = x.view(x.size(0),16*5*5)
# 		x = self.estimator(x)
# 		return x


def main():
    model = LeUNet()
    # model = torch.nn.DataParallel(model).cuda()\
    # model = model.cuda()
    model.load_state_dict(torch.load("../src/model_pm_china.pth"),strict=False)
    model.eval()
    # model = torch.quantization.convert(model)
    # model = torch.jit.script(model)
    # model = optimize_for_mobile(model)
    input_tensor = torch.rand(1,3,256,256)
    script_model = torch.jit.trace(model,input_tensor)
    script_model._save_for_lite_interpreter("mobile_model.pt")
    # model._save_for_lite_interpreter("mobile_model.ptl")


if __name__ == "__main__":
    main()