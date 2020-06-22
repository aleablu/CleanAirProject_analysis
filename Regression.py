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

import matplotlib.pyplot as plt


def parse_options():
    # parse command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save-model', action='store_true',
                        dest='save_model', default=False)
    parser.add_argument('--learning-rate', type=float, dest='lr', default=0.001)
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=32)
    parser.add_argument('--epochs', type=int, dest='epochs', default=20)
    parser.add_argument('--time-frame', type=str, dest='time', default='daily')
    parser.add_argument('-p', action='store_true',
                        dest='make_plots', default=False)
    parser.add_argument('-u', type=str, dest='model_to_use', default='none')
    parser.add_argument('--gap', type=int, dest='gap', default=50)
    args = parser.parse_args()
    return args


def plot_predictions(labels, predictions, title, num_data_to_plot, fname):
    x = range(0, num_data_to_plot)
    plt.clf()
    plt.plot(x, labels[:num_data_to_plot], label='original')
    plt.plot(x, predictions[:num_data_to_plot], label='predicted')
    plt.title(title)
    plt.legend()
    plt.savefig('plots/predictions_{}_{}epochs_{}bs_{}.png'.format(
                    TIME_FRAME, EPOCHS, BATCH_SIZE, fname))


def plot_tr_ts_loss(tr_losses, ts_losses):
    x = []
    for i in range(0, len(tr_losses)):
        x.append(i * GAP)
    plt.clf()
    plt.plot(x, tr_losses, label='train loss')
    plt.plot(x, ts_losses, label='test loss')
    plt.xlabel('epoch')
    plt.title('Train loss values, MSE')
    plt.legend()
    plt.savefig('plots/train_vs_test_loss_{}_{}epochs_{}bs_lr{}.png'.format(
                    TIME_FRAME, EPOCHS, BATCH_SIZE, LEARNING_RATE))


def plot_tr_ts_mae(maes):
    x = []
    for i in range(0, len(maes)):
        x.append(i * GAP)
    plt.clf()
    plt.plot(x, maes)
    plt.xlabel('epoch')
    plt.title('MAE value on test set during training')
    plt.savefig('plots/train_vs_test_mae_{}_{}epochs_{}bs_lr{}.png'.format(
                    TIME_FRAME, EPOCHS, BATCH_SIZE, LEARNING_RATE))


def load_data():
    # load data
    csv_path = 'data/merged_{}.csv'.format(TIME_FRAME)
    imgs_path = 'data/big_cells_images/resized_128'
    # init dataset class
    dataset = CleanAirDataset(csv_path, imgs_path)
    # split: train 80%, test 20%
    train_length = int(np.floor(len(dataset) * 0.8))
    # split data sequentially, no random
    train_set = tud.Subset(dataset, range(0, train_length))
    test_set = tud.Subset(dataset, range(train_length, len(dataset)))
    print('Train set: {} samples\nTest set:  {} samples'
          .format(len(train_set), len(test_set)))
    # init data loaders
    train_loader = tud.DataLoader(train_set, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=4)
    test_loader = tud.DataLoader(test_set, batch_size=1,
                                 shuffle=True, num_workers=4)
    return train_loader, test_loader, dataset


def test(loader):
    net.eval()
    loss = 0
    mae = 0
    orig, pred = [], []
    for batch_index, batch in enumerate(tqdm(loader)):
        imgs = batch['image'].to(device)
        weathers = batch['weather_data'].to(device)
        pms = batch['pm_label'].to(device)
        # make predictions
        predicted = net(imgs.float(), weathers)
        # denormalize values
        predicted = dataset.min_pm + pm_range * predicted
        pms = dataset.min_pm + pm_range * pms
        orig.append(pms.item())
        pred.append(predicted.item())
        # calc MSE value
        loss += criterion(pms.reshape(1, 1).float(), predicted).item()
        # calc MAE value
        mae += torch.abs(pms - predicted).item()
    loss /= len(loader)
    mae /= len(loader)
    net.train()
    return loss, orig, pred, mae


# parameters
args = parse_options()
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
EPOCHS = args.epochs
SAVE_MODEL = args.save_model
TIME_FRAME = args.time
MAKE_PLOTS = args.make_plots
MODEL_TO_USE = args.model_to_use
GAP = args.gap

train_loader, test_loader, dataset = load_data()

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
# init loss function
criterion = nn.MSELoss()
pm_range = dataset.max_pm - dataset.min_pm

if MODEL_TO_USE == 'none':
    # init optimizer for backpropagation
    optimizer = Adam(net.parameters(), lr=LEARNING_RATE)
    train_losses, tr_test_losses, tr_test_mae = [], [], []
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
            loss = criterion(predicted_pms,
                             pms.reshape(len(batch['pm_label']), 1).float())
            loss.backward()
            optimizer.step()
            # store loss value
            losses.append(loss.item())
        mean_loss = np.mean(losses)
        print('Mean loss = {}'.format(mean_loss))
        if epoch % GAP == 0:
            loss, _, _, mae = test(test_loader)
            print('MSE = {}, MAE = {}'.format(loss, mae))
            tr_test_losses.append(loss)
            tr_test_mae.append(mae)
            train_losses.append(mean_loss)

    print('\nTraining end')

    if SAVE_MODEL:
        time = datetime.now().strftime("%d-%m_%H-%M")
        model_fname = 'models/regressive_cnn_{}_{}.ptm'.format(time, TIME_FRAME)
        torch.save(net.state_dict(), model_fname)
        print('saving model in {}'.format(model_fname))

    if MAKE_PLOTS:
        plot_tr_ts_loss(train_losses, tr_test_losses)
        plot_tr_ts_mae(tr_test_mae)
        print('\nTrain vs test loss plot saved!')
else:
    # load already trained model
    net.load_state_dict(torch.load(MODEL_TO_USE))

print('\nTEST')
loss, orig, pred, mae = test(test_loader)
print('MSE = {}, MAE = {}'.format(loss, mae))


if MAKE_PLOTS:
    plot_predictions(np.array(orig), np.array(pred),
                     '{} data'.format(TIME_FRAME), 100, 'denormalized')
    print('\nPredictions plot saved!')
