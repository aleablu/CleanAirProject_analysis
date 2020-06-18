import torch
import pandas as pd
import numpy as np
import os
from torch.utils import data
from skimage import io
from torchvision import transforms


class CleanAirDataset(data.Dataset):

    def __init__(self, csv_path, imgs_path):
        self.imgs_path = imgs_path
        self.transform = transforms.ToTensor()
        df = pd.read_csv(csv_path)
        self.max_pm = df['num_particles'].max()
        self.min_pm = df['num_particles'].min()
        # normalize dataset
        for col in df.columns:
            max = df[col].max()
            min = df[col].min()
            if col not in ['time', 'cell_id']:
                df[col] = (df[col] - min) / (max - min)
        self.data_df = df

    # returns number of samples in this dataset
    def __len__(self):
        return self.data_df.shape[0]

    # returns idx-th sample, as image of the cell and it's weather data
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_fname = str(self.data_df.iloc[idx, 2]) + '.png'

        img_path = os.path.join(self.imgs_path, img_fname)
        image = io.imread(img_path)

        if self.transform:
            image = self.transform(image)

        weather = self.data_df.iloc[idx, [0, 3, 4, 5, 6, 7, 8, 9]]

        if type(weather['time']) == str:
            repl = {
                'Monday': 1, 'Thursday': 2, 'Wednesday': 3, 'Tuesday': 4,
                'Friday': 5, 'Saturday': 6, 'Sunday': 7, 'January': 1,
                'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10,
                'November': 11, 'December': 12, 'January': 1,
                'WIN': 1, 'SPR': 2, 'SUM': 3, 'AUT': 4
            }
            weather = weather.replace(repl)
        # convert weather data to column vector
        weather = np.array(weather).astype('float32').reshape(-1, 1)
        #print(weather)
        # convert to tensor
        weather = torch.from_numpy(weather)

        # get label, convert to to tensor then to column array
        pm_label = torch.from_numpy(np.asarray(self.data_df.iloc[idx, 1]))
        pm_label = pm_label.reshape(-1, 1)

        sample = {
                'image': image,
                'weather_data': weather,
                'pm_label': pm_label
        }
        return sample
