import torch
import torch.nn as nn
# import torchvision
import matplotlib.pyplot as plt
import numpy as np


class RegressiveCNN(nn.Module):

    def __init__(self):
        super(RegressiveCNN, self).__init__()
        # immagine rgb -> 3 canali input iniziale 12 out vuol dire
        # identificare 12 features
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()  # output.shape = 12x64x64
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # output.shape = 12x32x32

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()  # output.shape = 12x32x32
        self.pool2 = nn.MaxPool2d(kernel_size=3)  # output.shape = 12x10x10

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()  # output.shape = 24x10x10
        self.pool3 = nn.MaxPool2d(kernel_size=2)  # output.shape = 24x5x5

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()  # output.shape = 24x5x5
        self.pool4 = nn.MaxPool2d(kernel_size=2)  # output.shape = 24x2x2

        # qua comincia MLP, dopo layer di Flatten ho output.shape[1]*output.shape[2]*output.shape[3]
        # input features (vettore colonna) + 5 parametri meteo
        self.fc1 = nn.Linear(in_features=24*2*2 + 5, out_features=64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

    def show_image(self, img):
        npimg = img
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

    def forward(self, x, weather):
        output = self.conv1(x)
        output = self.relu1(output)
        output = self.pool1(output)
        # self.show_image(to_pil(output[0].cpu()))
        #print('pool1 --> {}'.format(output.shape))

        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pool2(output)
        # self.show_image(to_pil(output[0].cpu()))
        #print('pool2 --> {}'.format(output.shape))

        output = self.conv3(output)
        output = self.relu3(output)
        output = self.pool3(output)
        # self.show_image(to_pil(output[0].cpu()))
        #print('pool3 --> {}'.format(output.shape))

        output = self.conv4(output)
        output = self.relu4(output)
        output = self.pool4(output)
        # self.show_image(to_pil(output[0].cpu()))
        #print('pool4 --> {}'.format(output.shape))

        # trasformo in vettore colonna tramite layer di flatten sia la
        # feature map che i dati meteo

        output = torch.flatten(output, start_dim=1)
        weather = torch.flatten(weather, start_dim=1)
        #print(output.shape)
        # mi aspetto che weather sia un vettore colonna con i dati meteo
        output = torch.cat((output, weather), dim=1)

        output = self.fc1(output)
        output = self.dropout1(output)
        output = self.fc2(output)
        output = self.dropout2(output)
        output = self.fc3(output)

        return output
