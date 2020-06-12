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
        #self.pool1 = nn.MaxPool2d(kernel_size=2)  # output.shape = 12x32x32

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()  # output.shape = 24x64x64
        #self.pool2 = nn.MaxPool2d(kernel_size=2)  # output.shape = 24x32x32

        self.conv3 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()  # output.shape = 48x64x64
        #self.pool3 = nn.MaxPool2d(kernel_size=2)  # output.shape = 48x32x32

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()  # output.shape = 48x64x64
        self.pool4 = nn.MaxPool2d(kernel_size=2)  # output.shape = 48x32x32

        self.w1 = nn.Linear(in_features=8, out_features=32)
        self.w2 = nn.Linear(in_features=32, out_features=64)
        self.w3 = nn.Linear(in_features=64, out_features=64)
        self.w4 = nn.Linear(in_features=64, out_features=32)
        self.w5 = nn.Linear(in_features=32, out_features=8)
        
        # layer Bilinear, B(x1, x2) = x1^t * M * x2 + b
        # M, b imparati da Bilinear, x1=feature map
        # x2=dati, dati = dati_meteo + coord_cella + indice_temporale
        self.bilinear = nn.Bilinear(12*32*32, 8, 512)
        #self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(in_features=512, out_features=512)
        #self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=1)

    def show_image(self, img):
        npimg = img
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

    def forward(self, x, weather):
        output = self.conv1(x)
        output = self.relu1(output)
        #output = self.pool1(output)
        # self.show_image(to_pil(output[0].cpu()))
        # print('pool1 --> {}'.format(output.shape))

        output = self.conv2(output)
        output = self.relu2(output)
        #output = self.pool2(output)
        # self.show_image(to_pil(output[0].cpu()))
        # print('pool2 --> {}'.format(output.shape))

        output = self.conv3(output)
        output = self.relu3(output)
        #output = self.pool3(output)
        # self.show_image(to_pil(output[0].cpu()))
        # print('pool3 --> {}'.format(output.shape))

        output = self.conv4(output)
        output = self.relu4(output)
        output = self.pool4(output)
        # self.show_image(to_pil(output[0].cpu()))
        # print('pool4 --> {}'.format(output.shape))

        # trasformo in vettore colonna tramite layer di flatten sia la
        # feature map che i dati meteo
        output = torch.flatten(output, start_dim=1)
        weather = torch.flatten(weather, start_dim=1)

        # i dati meteo passano per un MLP separato, in modo da permettere
        # al modello di pesarli singolarmente e in modo accurato
        weather = self.w1(weather)
        weather = self.w2(weather)
        weather = self.w3(weather)
        weather = self.w4(weather)
        weather = self.w5(weather)

        # print(output.shape)
        # mi aspetto che weather sia un vettore colonna con i dati meteo
        #output = torch.cat((output, weather), dim=1)

        output = self.bilinear(output, weather)
        #output = self.dropout1(output)
        output = self.fc1(output)
        #output = self.dropout2(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        
        return output
