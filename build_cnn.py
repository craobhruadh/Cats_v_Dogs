import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

filepath = './train/'
batch_size = 128

# Transform all 3 channels (RGB) to be mean 0.5 and standard deviation 0.5
# or in other words between 0 and 1
transform = transforms.Compose([transforms.Resize(256),
	transforms.RandomCrop(256),
	transforms.Resize(128),
	transforms.ToTensor(),
	transforms.Normalize((0.5,0.5, 0.5), (0.5,0.5, 0.5))])



train_data = torchvision.datasets.ImageFolder(filepath, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size= batch_size, shuffle=True)
