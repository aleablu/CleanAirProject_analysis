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

import matplotlib.pyplot as plt


def plot_predictions(labels, predictions, title, num_data_to_plot, fname):
    x = range(0, num_data_to_plot)
    plt.clf()
    plt.plot(x, labels[:num_data_to_plot], label='original')
    plt.plot(x, predictions[:num_data_to_plot], label='predicted')
    plt.title(title)
    plt.legend()
    plt.savefig('plots/predictions_{}_{}epochs_{}bs_{}.png'.format(TIME_FRAME, EPOCHS, BATCH_SIZE, fname))


# parse command line options
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save-model', action='store_true',
                    dest='save_model', default=False)
parser.add_argument('--learning-rate', type=float, dest='lr', default=0.001)
parser.add_argument('--batch-size', type=int, dest='batch_size', default=128)
parser.add_argument('--epochs', type=int, dest='epochs', default=20)
parser.add_argument('--time-frame', type=str, dest='time', default='daily')
parser.add_argument('-p', action='store_true',
                    dest='make_plots', default=False)

args = parser.parse_args()
# parameters
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
EPOCHS = args.epochs
SAVE_MODEL = args.save_model
TIME_FRAME = args.time
MAKE_PLOTS = args.make_plots

# load data
csv_path = 'data/merged_{}.csv'.format(TIME_FRAME)
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
        loss = criterion(predicted_pms, pms.reshape(len(batch['image']), 1).float())
        loss.backward()
        optimizer.step()

        # store loss value
        losses.append(float(loss))
    mean_loss = np.mean(losses)
    print('Mean loss = {}'.format(mean_loss))
    train_losses.append(mean_loss)

print('\nTraining end')

if SAVE_MODEL:
    time = datetime.now().strftime("%d_%m_%y_%H_%M")
    model_fname = 'models/regressive_cnn_' + time + '.ptm'
    torch.save(net.state_dict(), model_fname)
    print('saving model in {}'.format(model_fname))

print('\nBegin testing!')
mse_total = 0
orig, pre = [], []
orig_de, pre_de = [], []
for batch_index, batch in enumerate(tqdm(test_loader)):
    imgs = batch['image'].to(device)
    weathers = batch['weather_data'].to(device)
    pms = batch['pm_label'].to(device)
    # make predictions and store label + predicted
    orig.append(torch.flatten(pms.cpu().detach()))
    predicted = net(imgs.float(), weathers)
    pre.append(torch.flatten(predicted.cpu().detach()))
    # denormalize values, store them 
    predicted = dataset.min_pm + pm_range * predicted
    pms = dataset.min_pm + pm_range * pms
    orig_de.append(torch.flatten(pms.cpu().detach()))
    pre_de.append(torch.flatten(predicted.cpu().detach()))
    # calc loss function MSE
    mse = 0
    for i in range(len(pms)):
        mse += (pms[i] - predicted[i]) ** 2
    mse /= len(pms)
    mse_total += mse
mse_total /= len(test_loader)

print('mean MSE value on test data = {}\n'.format(float(mse_total)))
if MAKE_PLOTS:
    plot_predictions(np.array(orig), np.array(pre), '{} normalized data'.format(TIME_FRAME), 100, 'norm')
    plot_predictions(np.array(orig_de), np.array(pre_de), '{} denormalized data'.format(TIME_FRAME), 100, 'denorm')
    print('\nPredictions plot saved!')
