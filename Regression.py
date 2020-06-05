import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as tud

import numpy as np
from datetime import datetime
import argparse

from Network import RegressiveCNN
from CleanAirDataset import CleanAirDataset
from tqdm import tqdm
from torchvision import transforms

import plotly.express as px

# parse command line options
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save-model', action='store_true',
                    dest='save_model', default=False)
parser.add_argument('--learning-rate', type=float, dest='lr', default=0.001)
parser.add_argument('--batch-size', type=int, dest='batch_size', default=128)
parser.add_argument('--epochs', type=int, dest='epochs', default=20)

args = parser.parse_args()
# parameters
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
EPOCHS = args.epochs
SAVE_MODEL = args.save_model
# load data
csv_path = 'data/merged_seasonally.csv'
imgs_path = 'data/cells_images/resized_64'
# transforms img in Tensor, backend uses Pillow that normalizes img in [0,1]
transform = transforms.ToTensor()
dataset = CleanAirDataset(csv_path, imgs_path, transform=transform)

# split: train 80%, test 20%
train_length = int(np.floor(len(dataset) * 0.8))
test_length = len(dataset) - train_length
# split data sequentially, no random
train = tud.Subset(dataset, range(0, train_length))
test = tud.Subset(dataset, range(train_length, len(dataset)))
print(len(train), len(test))
# init data loaders
train_loader = tud.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = tud.DataLoader(test, batch_size=1, shuffle=True, num_workers=4)


# If possible runs on GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('RUNNING ON {}'.format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print("-- RUNNING ON CPU --")

# init network
net = RegressiveCNN()
net.to(device)

# init loss function and optimizer for backpropagation
criterion = nn.MSELoss()
optimizer = Adam(net.parameters(), lr=LEARNING_RATE)
pm_range = dataset.max_pm - dataset.min_pm
train_losses = []
for epoch in range(EPOCHS):
    print('Epoch {}'.format(epoch))
    losses = []
    for batch_index, batch in enumerate(tqdm(train_loader)):
        # get image, weather params and pm label and move these
        # tensors to gpu if available
        imgs = batch['image'].to(device)
        weathers = batch['weather_data'].to(device)
        pms = batch['pm_label'].to(device)

        # set parameter's gradients to zero
        optimizer.zero_grad()

        # get output from network
        predicted_pms = net(imgs.float(), weathers)

        # denormalize predicted and label pms
        predicted_pms = dataset.min_pm + pm_range * predicted_pms
        pms = dataset.min_pm + pm_range * pms
        
        # calc loss and backpropagate
        loss = criterion(predicted_pms, pms.reshape(len(batch['pm_label']), 1).float())
        loss.backward()
        optimizer.step()

        # store loss value
        losses.append(float(loss))
    mean_loss = np.mean(losses)
    print('Mean loss = {}'.format(mean_loss))
    train_losses.append(mean_loss)

print('\nTraining end')
fig = px.line(x=range(0, EPOCHS), y=train_losses,
              labels={'x': 'epochs', 'y': 'MSE loss'})
fig.write_image('plots/train_loss_bs' + str(BATCH_SIZE) + '_lr'
                + str(LEARNING_RATE) + '.png')
print('train losses saved in plots')

if SAVE_MODEL:
    time = datetime.now().strftime("%d_%m_%y_%H_%M")
    model_fname = 'models/regressive_cnn_' + time + '.ptm'
    torch.save(net.state_dict(), model_fname)
    print('saving model in {}'.format(model_fname))

print('\nBegin testing!')
mse_total = 0
for batch_index, batch in enumerate(tqdm(test_loader)):
    imgs = batch['image'].to(device)
    weathers = batch['weather_data'].to(device)
    pms = batch['pm_label'].to(device)

    predicted = net(imgs.float(), weathers)
    predicted = dataset.min_pm + pm_range * predicted
    pms = dataset.min_pm + pm_range * pms
    mse = 0
    for i in range(len(pms)):
        mse += (pms[i] - predicted[i]) ** 2
    mse /= len(pms)
    mse_total += mse
mse_total /= len(test_loader)

print('mean MSE value on test data = {}'.format(float(mse_total)))
