import os
import cv2
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

READ_DATA = False

class catsdogs():

	img_size = 50
	cats = "PetImages/Cat"
	dogs = "PetImages/Dog"

	labels = {cats: 0, dogs: 1}

	training_data = []

	counts = [0,0]

	def produce_data(self):
		
		for label in self.labels:
			print("Importing " + str(label))
			for f in tqdm(os.listdir(label)):
				if "jpg" in f:
					try:
						# import data
						path = os.path.join(label, f)
						img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
						
						# preprocess data
						img = cv2.resize(img, (self.img_size, self.img_size))
						self.training_data.append([np.array(img), np.eye(2)[self.labels[label]]])

						if label == self.cats:
							self.counts[0] += 1
						else:
							self.counts[1] += 1
					
					except Exception as e:
						pass

		# randomize dataset
		np.random.shuffle(self.training_data)

		np.save("training_data.npy", self.training_data)

		print("-- DATASET BALANCE --")
		print("Total imgs -->", self.counts[0]+self.counts[1])
		print("Cats imgs --> ", self.counts[0])
		print("Dogs imgs --> ", self.counts[1])




class Net(nn.Module):
	
	def __init__(self):
		super().__init__()

		# convolutional layers
		self.conv1 = nn.Conv2d(1, 32, 5)
		self.conv2 = nn.Conv2d(32, 64, 5)
		self.conv3 = nn.Conv2d(64, 128, 5)

		# passes data in the conv layer finding final dim
		x = torch.randn(50,50).view(-1, 1, 50, 50)
		self._to_linear = None
		self.convs(x)

		# dense layers
		self.dense1 = nn.Linear(self._to_linear, 512)
		self.dense2 = nn.Linear(512, 2)


	def convs(self, x):

		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
		x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

		if self._to_linear is None:
			self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

		return x

	
	def forward(self, x):
		''' Defines how data is passing in the netowork '''

		# applies convolution and pooling
		x = self.convs(x)
		
		# resizes data
		x = x.view(-1, self._to_linear)

		# applies dense layers
		x = F.relu(self.dense1(x))
		x = self.dense2(x)

		# output layer
		x = F.softmax(x, dim=1)

		return x

# If possible runs on GPU
if torch.cuda.is_available():
	device = torch.device("cuda:0")
	print("-- RUNNING ON THE GPU --")
else:
	device = torch.device("cpu")
	print("-- RUNNING ON THE CPU --")

if READ_DATA:
	catsdogs().produce_data()

training_data = np.load("training_data.npy", allow_pickle=True)
print("-- DATASET LOADED --")

net = Net().to(device)

# init optimizer and loss func
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_func = nn.MSELoss()

# create data tensors
X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50) / 255.0
y = torch.Tensor([i[1] for i in training_data])

# split train and test
split_size = 0.1
val_size = int(len(X) * split_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

def fwd_pass(X, y, train=False):

	if train:
		net.zero_grad()
	
	outputs = net(X)

	matches  = [torch.argmax(i)==torch.argmax(j) for i, j in zip(outputs, y)]
	in_sample_acc = matches.count(True)/len(matches)

	acc = matches.count(True)/len(matches)
	loss = loss_func(outputs, y)

	if train:
		loss.backward()
		optimizer.step()

	return acc, loss


def train(net):

	# training params
	batch_size = 8
	epochs = 10

	for epoch in range(epochs):
		print("Epoch ", epoch)
		for i in tqdm(range(0, len(train_X), batch_size)):
			
			batch_X = train_X[i:i+batch_size].view(-1, 1, 50, 50)
			batch_y = train_y[i:i+batch_size]

			# loads current batch into memory
			batch_X, batch_y = batch_X.to(device), batch_y.to(device)

			# pass in network
			acc, loss = fwd_pass(batch_X,batch_y, train=True)

		print(f"Acc: {round(float(acc),2)}  Loss: {round(float(loss),4)}")

def train2(net, logfile):

	name = f"model-{int(time.time())}"
	print("MODEL NAME --> ", name)
	f = open(logfile, "a")

	# training params
	batch_size = 32
	epochs = 10

	for epoch in range(epochs):
		print("Epoch ", epoch)
		epoch_acc = []
		epoch_loss = []
		for i in tqdm(range(0, len(train_X), batch_size)):
			
			batch_X = train_X[i:i+batch_size].view(-1, 1, 50, 50)
			batch_y = train_y[i:i+batch_size]

			# loads current batch into memory
			batch_X, batch_y = batch_X.to(device), batch_y.to(device)

			# pass in network
			acc, loss = fwd_pass(batch_X,batch_y, train=True)

			epoch_acc.append(float(acc))
			epoch_loss.append(float(loss))

			f.write(f"{name},{round(time.time(),3)},in_sample,{round(float(acc),2)},{round(float(loss),4)}\n")

		print(f"Acc: {round(float(np.mean(epoch_acc)),2)}  Loss: {round(float(np.mean(epoch_loss)),4)}")

	f.close()

def test(net, test_X, test_y, logfile):
	epoch_acc = []
	epoch_loss = []

	name = f"model-{int(time.time())}"
	print("MODEL NAME --> ", name)
	f = open(logfile, "a")
	f.write(f"model_name,time,sample_type,accuracy,loss\n")

	for i in tqdm(range(0, len(test_X), 2)):
			
		x = test_X[i:i+2].view(-1, 1, 50, 50)
		y = test_y[i:i+2]
		
		# loads current batch into memory
		x, y = x.to(device), y.to(device)
		
		# pass in network
		acc, loss = fwd_pass(x,y, train=False)
		
		epoch_acc.append(float(acc))
		epoch_loss.append(float(loss))

		f.write(f"{name},{round(time.time(),3)},test_sample,{round(float(acc),2)},{round(float(loss),4)}\n")

	print(f"Acc: {round(float(np.mean(epoch_acc)),2)}  Loss: {round(float(np.mean(epoch_loss)),4)}")

TRAIN = False

if TRAIN:
	train2(net, "model4")


	with open('net2', 'wb') as ser:
		pickle.dump(net, ser)

with open('net2', 'rb') as ser:
	net = pickle.load(ser)

test(net, test_X, test_y, "test1")