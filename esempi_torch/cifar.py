import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
	def __init__(self):

		super(Net, self).__init__()

		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool1 = nn.MaxPool2d(2, 2)

		# on tutorial another conv/pool layer

		self.fc1 = nn.Linear(6*14*14, 60)
		self.fc2 = nn.Linear(60, 10)

	def forward(self, x):

		x = self.conv1(x)
		x = self.pool1(x)

		x = x.view(-1, 6*14*14)
	
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)

		return x

# If possible runs on GPU
if torch.cuda.is_available():
	device = torch.device("cuda:0")
	print("-- RUNNING ON THE GPU --")
else:
	device = torch.device("cpu")
	print("-- RUNNING ON THE CPU --")

net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

epochs = 20

for epoch in range(epochs):

	print("Epoch ", epoch)
	running_loss = []

	for i, data in enumerate(tqdm(trainloader), 0):

		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data[0].to(device), data[1].to(device)


		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss.append(float(loss))

	print("Loss --> ", float(np.mean(running_loss)))
	print()

with open('net', 'wb') as ser:
	pickle.dump(net, ser)
