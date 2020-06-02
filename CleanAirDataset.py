import torch
import pandas as pd
import numpy as np
import os
from torch.utils import data
from skimage import io


class CleanAirDataset(data.Dataset):

    def __init__(self, csv_path, imgs_path, transform=None):
        self.data_df = pd.read_csv(csv_path)
        self.imgs_path = imgs_path
        self.transform = transform

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

        # swap img dimensions, to fit torch standard
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        # convert numpy image to tensor
        image = torch.from_numpy(image)

        weather = self.data_df.iloc[idx, 3:]
        # convert weather data to column vector
        weather = np.array(weather).astype('float32').reshape(-1, 1)
        # convert to tensor
        weather = torch.from_numpy(weather)

        # get label, convert to to tensor then to column array
        pm_label = torch.from_numpy(np.asarray(self.data_df.iloc[idx, 1]))
        pm_label = pm_label.reshape(-1, 1)
        #print(image)
        if self.transform:
            image = self.transform(image)
            #weather = self.transform(weather)
            #pm_label = self.transform(pm_label)
        #print(image)
        sample = {
                'image': image,
                'weather_data': weather,
                'pm_label': pm_label
        }

      
        return sample
