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

from sklearn.metrics import r2_score


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


def plot_test_during_train(d):
    x = []
    for i in range(0, len(d['rmse'])):
        x.append(i * GAP)
    plt.clf()
    plt.plot(x, train_rmse, label='train RMSE')
    plt.plot(x, d['rmse'], label='test RMSE')
    plt.xlabel('epoch')
    plt.title('RMSE value on test set during training, {}'.format(TIME_FRAME))
    plt.legend()
    plt.savefig('plots/train_vs_test_RMSE_{}_{}epochs_{}bs_lr{}.png'.format(
                    TIME_FRAME, EPOCHS, BATCH_SIZE, LEARNING_RATE))
    plt.clf()
    plt.plot(x, train_r2, label='train r2_Score')
    plt.plot(x, d['r2'], label='test r2_Score')
    plt.xlabel('epoch')
    plt.title('r^2-score value on test set during training, {}'.format(TIME_FRAME))
    plt.legend()
    plt.savefig('plots/train_vs_test_r2_{}_{}epochs_{}bs_lr{}.png'.format(
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


def test(loader, training=False):
    # set the network in evaluation mode
    net.eval()
    rmse, r2 = 0, 0
    orig, pred = [], []
    for batch_index, batch in enumerate(tqdm(loader)):
        # split batch and send it to the gpu if available
        imgs = batch['image'].to(device)
        weathers = batch['weather_data'].to(device)
        pms = batch['pm_label'].to(device)
        # make predictions
        predicted = net(imgs.float(), weathers)
        # denormalize values
        predicted = dataset.min_pm + pm_range * predicted
        pms = dataset.min_pm + pm_range * pms
        # store original and predicted data
        orig.append(pms.item())
        pred.append(predicted.item())
        # calc RMSE value
        rmse += criterion(pms.reshape(1, 1).float(), predicted).item() ** 0.5
    # calc mean RMSE on the test set
    rmse /= len(loader)
    # calc r2 score between original and predicted data, denormalized
    r2 = r2_score(orig, pred)
    # set network back to train mode, this method is called while training
    if training:
        net.train()
    # return original and predicted data, and metrics
    return orig, pred, rmse, r2


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
#criterion = nn.MSELoss()
criterion = nn.L1Loss()
pm_range = dataset.max_pm - dataset.min_pm

if MODEL_TO_USE == 'none':
    # init optimizer for backpropagation
    optimizer = Adam(net.parameters(), lr=LEARNING_RATE)
    train_rmse, train_r2 = [], []
    test_during_train = {'rmse': [], 'r2': []}
    for epoch in range(EPOCHS):
        print('Epoch {}'.format(epoch))
        rmses, r2s = [], []
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
            rmses.append(loss.item())
            r2s.append(r2_score(pms.reshape(len(batch['pm_label']), 1).cpu().detach().numpy(),
                       predicted_pms.reshape(len(batch['pm_label']), 1).cpu().detach().numpy()))
        mean_rmse = np.mean(rmses)
        mean_r2 = np.mean(r2s)
        print('RMSE = {}, r2_score = {}'.format(mean_rmse, mean_r2))
        if epoch % GAP == 0:
            _, _, rmse, r2 = test(test_loader, training=True)
            print('TEST --> RMSE = {}, r2_score = {}'.format(rmse, r2))
            test_during_train['rmse'].append(rmse)
            test_during_train['r2'].append(r2)
            train_rmse.append(mean_rmse)
            train_r2.append(mean_r2)

    print('\nTraining end')
    if SAVE_MODEL:
        time = datetime.now().strftime("%d-%m_%H-%M")
        model_fname = 'models/regressive_cnn_{}_{}.ptm'.format(time, TIME_FRAME)
        torch.save(net.state_dict(), model_fname)
        print('saving model in {}'.format(model_fname))
    if MAKE_PLOTS:
        plot_test_during_train(test_during_train)
        print('\nMetrics over test set during train plot saved!')
else:
    # load already trained model
    net.load_state_dict(torch.load(MODEL_TO_USE))

print('\nTEST')
orig, pred, rmse, r2 = test(test_loader)

if MAKE_PLOTS:
    plot_predictions(np.array(orig), np.array(pred),
                     '{} data'.format(TIME_FRAME), 100, 'denormalized')
    print('\nPredictions plot saved!')

print('RMSE = {}, r2_score = {}'.format(rmse, r2))