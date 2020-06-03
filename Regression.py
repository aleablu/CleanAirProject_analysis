import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as tud

import numpy as np

from Network import RegressiveCNN
from CleanAirDataset import CleanAirDataset
from tqdm import tqdm
from torchvision import transforms

# parameters
BATCH_SIZE = 128

# load data
csv_path = 'data/merged_daily.csv'
imgs_path = 'data/cells_images/resized_64'
normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
transform = transforms.Compose(
        [transforms.ToTensor(),  # transform to tensor --> [0,1]
            normalize])  # transform with mean and std 0.5 --> [-1, 1]
dataset = CleanAirDataset(csv_path, imgs_path, transform=transform)

# split: train 80%, test 20%
train_length = int(np.floor(len(dataset) * 0.8))
test_length = len(dataset) - train_length
train, test = tud.random_split(dataset, [train_length, test_length])

# init data loaders
train_loader = tud.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = tud.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


# If possible runs on GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('running on {}'.format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print("-- RUNNING ON THE CPU --")

# init network
net = RegressiveCNN()
net.to(device)

# init loss function and optimizer for backpropagation
criterion = nn.MSELoss()
optimizer = Adam(net.parameters(), lr=0.001)

epochs = 10000
for epoch in range(epochs):
    print('Epoch {}'.format(epoch))
    losses = []
    for batch_index, batch in enumerate(tqdm(train_loader)):
        # get image, weather params and pm label and move these
        # tensors to gpu if available
        imgs = batch['image'].to(device)
        weathers = batch['weather_data'].to(device)
        pms = batch['pm_label'].to(device)
        #print(imgs)
        # set parameter's gradients to zero
        optimizer.zero_grad()

        # get output from network
        predicted_pms = net(imgs.float(), weathers)
        # calc loss and backpropagate
        loss = criterion(predicted_pms, pms.reshape(len(batch['pm_label']), 1).float())
        loss.backward()
        optimizer.step()

        # store loss value
        losses.append(float(loss))
    print('Mean loss = {}'.format(np.mean(losses)))
