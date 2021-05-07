#!/usr/bin/python

import os
import torch,cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils, datasets
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.models as models
import time,sys




def image_load(imgPath, img_size):
	##### image load and resize with limited-padding

	def pad(image, min_height, min_width):
		h, w, d = image.shape

		if h < min_height:
			h_pad_top = int((min_height - h) / 2.0)
			h_pad_bottom = min_height - h - h_pad_top
		else:
			h_pad_top = 0
			h_pad_bottom = 0

		if w < min_width:
			w_pad_left = int((min_width - w) / 2.0)
			w_pad_right = min_width - w - w_pad_left
		else:
			w_pad_left = 0
			w_pad_right = 0

		return cv2.copyMakeBorder(image, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, cv2.BORDER_CONSTANT,value=(255, 255, 255))

	def resize_to_square(image, size):
		h, w, d = image.shape
		ratio = size / max(h, w)
		resized_image = cv2.resize(image, (int(w * ratio), int(h * ratio)), cv2.INTER_AREA)
		return resized_image


	try:
		# print(f" image_load {imgPath } ...")
		src = cv2.imread(imgPath)
		img_resized = pad(resize_to_square(src, img_size), img_size, img_size)
		pil_img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
		return pil_img

	except:
		print(f"图片读书失败: {imgPath}")
		return None


class ImageNetDataset(Dataset):
	# loads data from the csv file

	def __init__(self, img_size, csv_file, root_dir, transform=None):
		self.imagenet_frame = pd.read_csv(csv_file)
		self.img_size = img_size
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.imagenet_frame)

	def __getitem__(self, idx):
		imgid = torch.tensor([int(self.imagenet_frame.iloc[idx, 0])])
		query_name = self.root_dir + self.imagenet_frame.iloc[idx, 2]
		inclass_name = os.path.join(self.root_dir, self.imagenet_frame.iloc[idx, 3])
		outclass_name = os.path.join(self.root_dir, self.imagenet_frame.iloc[idx, 4])


		query_image = image_load(query_name, self.img_size) ### PIL RGB
		inclass_image = image_load(inclass_name, self.img_size)
		outclass_image = image_load(outclass_name, self.img_size)


		if self.transform:
			query_image = self.transform(query_image)
			inclass_image = self.transform(inclass_image)
			outclass_image = self.transform(outclass_image)

		sample = [imgid, query_name, query_image, inclass_image, outclass_image]
		return sample



def resnet101(**kwargs):
	model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [3, 4, 23, 3])
	model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
	return model




def train_model(sampleCSV, es, bs, datasetPath, num_worker=8):

	transformations = transforms.Compose([
		# transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

	# defining data loader objects
	print("数据加载 ... ")
	img_size = 64
	trainset = ImageNetDataset(img_size, csv_file=sampleCSV, root_dir=datasetPath,transform=transformations)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=num_worker)


	net = models.resnet101(pretrained=False)
	# resetting fully connected layer
	num_ftrs = net.fc.in_features
	net.fc = nn.Linear(num_ftrs, 4096)
	net.to(device)


	# hyperparameters
	epochs = es
	train_ls = 0.001
	iterm_log = 20
	min_loss = 100
	criterion = nn.TripletMarginLoss(margin=1.0, p=2)
	# optimizer = optim.Adam(net.parameters(), lr = 0.001)
	optimizer = optim.SGD(net.parameters(), lr=train_ls, momentum=0.9)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
	losses_epoch = np.zeros(epochs)

	best_info = None
	for epoch in range(epochs):
		start = time.time()
		loss_sum = 0
		for i, data in enumerate(trainloader):
			imgid, query_name, query, inclass, outclass = data
			Q = F.interpolate(query, scale_factor=3.5)
			P = F.interpolate(inclass, scale_factor=3.5)
			N = F.interpolate(outclass, scale_factor=3.5)
			Q, P, N = Q.to(device), P.to(device), N.to(device)
			optimizer.zero_grad()

			# forward - obtain outputs from neural net
			Q_OUT = net(Q)
			P_OUT = net(P)
			N_OUT = net(N)

			# backward + optimize
			loss = criterion(Q_OUT, P_OUT, N_OUT)
			loss.backward()
			loss_sum = loss_sum + loss.item()
			optimizer.step()

			if i % iterm_log == 0:
				print('loss at iteration {}: {}'.format(i, loss.item()))

		losses_epoch[epoch] = loss_sum
		end = time.time()
		tim_cons = (end - start) / 60


		if min_loss > loss_sum:

			min_loss = loss_sum
			tmp_model = "checkpoint_best.ckpt"
			torch.save(net.state_dict(), tmp_model)
			best_info = f'epoch_{epoch}'
			print(f"Save best at epoch of {epoch} with total-loss: {loss_sum} ")


		print(f" Epoch: {str(epoch)} / {str(epochs)}  total-loss: {loss_sum}  average-loss: {loss_sum/(i+1)} time-cost: {tim_cons} ")

		scheduler.step()
		torch.save(net.state_dict(), 'checkpoint.ckpt')
		torch.save(net, 'full_model.ckpt')

	print(f'Finished Training, total-loss: {loss_sum} , best_info: {best_info}')
	# np.savetxt('losses.csv', losses_epoch, delimiter=',')
	df_loss = pd.DataFrame([losses_epoch]).T
	df_loss.to_csv("losses.csv")




if __name__ == "__main__":
	s = time.time()
	epochs = 100
	batch_size = 30
	# sampleCSV = './file_table.csv'
	# datasetPath = '../data/tiny-imagenet-200/train/'

	sampleCSV = sys.argv[1]
	datasetPath = sys.argv[2]

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	train_model(sampleCSV, epochs, batch_size, datasetPath)

	print(f'训练耗时: {time.time() - s }')


