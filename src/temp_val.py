import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from train_pm import Dataset, double_conv, LeUNet, val


if __name__ == '__main__':

	data = pd.read_csv('../final_data.csv')
	data_train = data.sample(frac=0.8,random_state=17)
	data_val = data.loc[~data.index.isin(data_train.index)]
	files_val = list(data_val['filename'])
	ppm_val = list(data_val['ppm'])
	ids_val = [i for i in range(len(files_val))]
	
	model = LeUNet()
	model = torch.nn.DataParallel(model).cuda()
	model.load_state_dict(torch.load("model_hazy_best.pth"),strict=False) # on GPU
	criterion = nn.MSELoss().cuda()
	optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
	
	val_dataset = Dataset(ids_val, files_val, ppm_val, transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize(mean=[0.5231, 0.5180, 0.5115],std=[0.2014, 0.2018, 0.2100]),]))
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
	
	val_loss = val(val_loader, model, criterion)
	val_rmse = np.sqrt(val_loss)
	print('Validation RMSE: {:.4f}'.format(val_rmse))