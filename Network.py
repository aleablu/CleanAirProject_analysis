import torch
import torch.nn as nn


class RegressiveCNN(nn.Module):

    def __init__(self):
        super(RegressiveCNN, self).__init__()
        # LeNet structure!

        # immagine rgb -> 3 canali input iniziale 12 out vuol dire
        # identificare 12 features
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,
                               stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 3x64x64

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=6)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 6x32x32

        # self.conv3 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(num_features=24)  # output.shape = 24x32x32
        # self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=2)

        # self.conv4 = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(num_features=12)  # output.shape = 12x16x16
        # self.relu4 = nn.ReLU()
        # self.pool4 = nn.MaxPool2d(kernel_size=2)

        # weather MLP
        self.w1 = nn.Linear(in_features=8, out_features=32)
        self.w_bn1 = nn.BatchNorm1d(num_features=32)
        self.w2 = nn.Linear(in_features=32, out_features=64)
        self.w_bn2 = nn.BatchNorm1d(num_features=64)
        self.w3 = nn.Linear(in_features=64, out_features=128)
        self.w_bn3 = nn.BatchNorm1d(num_features=128)
        self.w4 = nn.Linear(in_features=128, out_features=64)
        self.w_bn4 = nn.BatchNorm1d(num_features=64)
        self.w5 = nn.Linear(in_features=64, out_features=8)

        # layer Bilinear, B(x1, x2) = x1^t * M * x2 + b
        # M, b imparati da Bilinear, x1=feature map
        # x2=dati, dati = dati_meteo + coord_cella + indice_temporale
        self.bilinear = nn.Bilinear(6*32*32, 8, 256)
        self.bilin_bn = nn.BatchNorm1d(num_features=256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.mlp_bn1 = nn.BatchNorm1d(num_features=128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.mlp_bn2 = nn.BatchNorm1d(num_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.mlp_bn3 = nn.BatchNorm1d(num_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=16)
        self.mlp_bn4 = nn.BatchNorm1d(num_features=16)
        self.fc5 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x, weather):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool1(output)
        # print('pool1 --> {}'.format(output.shape))

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.pool2(output)
        # print('pool2 --> {}'.format(output.shape))

        # output = self.conv3(output)
        # output = self.bn3(output)
        # output = self.relu3(output)
        # output = self.pool3(output)
        # print('pool3 --> {}'.format(output.shape))

        # output = self.conv4(output)
        # output = self.bn4(output)
        # output = self.relu4(output)
        # output = self.pool4(output)
        # print('pool4 --> {}'.format(output.shape))

        # trasformo in vettore colonna tramite layer di flatten sia la
        # feature map che i dati meteo
        output = torch.flatten(output, start_dim=1)
        weather = torch.flatten(weather, start_dim=1)

        # MLP WEATHER
        weather = self.w1(weather)
        weather = self.w_bn1(weather)
        weather = self.w2(weather)
        weather = self.w_bn2(weather)
        weather = self.w3(weather)
        weather = self.w_bn3(weather)
        weather = self.w4(weather)
        weather = self.w_bn4(weather)
        weather = self.w5(weather)

        # MLP FINALE
        output = self.bilinear(output, weather)
        output = self.bilin_bn(output)
        output = self.dropout1(output)
        output = self.fc1(output)
        output = self.mlp_bn1(output)
        output = self.dropout2(output)
        output = self.fc2(output)
        output = self.mlp_bn2(output)
        output = self.fc3(output)
        output = self.mlp_bn3(output)
        output = self.fc4(output)
        output = self.mlp_bn4(output)
        output = self.fc5(output)

        return output
