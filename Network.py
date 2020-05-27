import torch
import torch.nn as nn


class RegressiveCNN(nn.Module):

    def __init__(self):
        super(RegressiveCNN, self).__init__()
        # immagine rgb -> 3 canali input iniziale (ma caricandole con io e
        # portandole in tensori diventano 4 ?!) 12 out vuol dire identificare
        # 12 features
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        # altro layer di convoluzione, 12 in 12 out
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # maxpool, riduce dimensione immagine prendendo val max in kernel 2x2
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        # layer finale di convoluzione, 24 output channels, facendo maxpool
        # con kernel 2x2 ottengo immagini rappresentanti le feature estratte
        # di dimensione 16*16
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        # qua comincia MLP, dopo layer di Flatten ho 128 * 128 * 24
        # + 5 parametri meteo
        self.fc1 = nn.Linear(in_features=128 * 128 * 24 + 5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x, weather):
        output = self.conv1(x)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.pool(output)

        output = self.conv3(output)
        output = self.relu3(output)

        output = self.conv4(output)
        output = self.relu4(output)

        # trasformo in vettore colonna tramite layer di flatten sia la
        # feature map che i dati meteo
        output = torch.flatten(output, start_dim=1)
        weather = torch.flatten(weather, start_dim=1)

        # mi aspetto che weather sia un vettore colonna con i dati meteo
        output = torch.cat((output, weather), dim=1)

        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)

        return output
