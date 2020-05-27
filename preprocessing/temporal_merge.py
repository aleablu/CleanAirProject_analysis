import pandas as pd
import os
import re
from tqdm import tqdm


def merge(files_list, offset, time):
    sample = {
        'time': '', 'num_particles': 0, 'cell_id': 0,
        'temperature': 0, 'atm_pressure': 0,
        'humidity': 0, 'wind_direction': 0, 'wind_speed': 0
    }
    cols = ['time', 'num_particles', 'cell_id',
            'temperature', 'atm_pressure', 'humidity',
            'wind_direction', 'wind_speed']
    df = pd.DataFrame(columns=cols)
    for csv in tqdm(files_list):
        temp = pd.read_csv(os.path.join(cells_csv_path, csv))
        cell_id = int(csv[:-offset])
        for index, row in temp.iterrows():
            sample['time'] = row['time']
            sample['num_particles'] = row['num_particles']
            sample['cell_id'] = cell_id
            sample['temperature'] = row['temperature']
            sample['atm_pressure'] = row['atm_pressure']
            sample['humidity'] = row['humidity']
            sample['wind_direction'] = row['wind_direction']
            sample['wind_speed'] = row['wind_speed']
            df = df.append(sample, ignore_index=True)
    print('Saving merged csv')
    print(df)
    df.to_csv('../data/merged_' + time + '.csv', index=False)


cells_csv_path = '../data/cells_samples_csv'
cells_imgs_path = '../data/cells_images'

# build lists of files for each time period
daily, weekly, monthly, seasonally = [], [], [], []

p_d = re.compile('[0-9]+_daily.csv')
p_w = re.compile('[0-9]+_weekly.csv')
p_m = re.compile('[0-9]+_monthly.csv')
p_s = re.compile('[0-9]+_seasonally.csv')

for f in os.listdir(cells_csv_path):
    if p_d.match(f):
        daily.append(f)
    elif p_w.match(f):
        weekly.append(f)
    elif p_m.match(f):
        monthly.append(f)
    elif p_s.match(f):
        seasonally.append(f)

# merge files and save them
print('Merging daily files!')
merge(daily, 10, 'daily')
print('Merging weekly files!')
merge(weekly, 11, 'weekly')
print('Merging monthly files!')
merge(monthly, 12, 'monthly')
print('Merging seasonally files!')
merge(seasonally, 15, 'seasonally')
