import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
from datetime import datetime


def group_daily(data):
    res = data.copy()
    res = res.groupby(lambda x: x.day_name()).mean()
    return res


def group_weekly(data):
    res = data.copy()
    res = res.groupby(lambda x: x.weekofyear).mean()
    return res


def group_monthly(data):
    res = data.copy()
    res = res.groupby(lambda x: x.month_name()).mean()
    return res


def group_seasonally(data):
    seasons = {
        1: 'WIN', 2: 'WIN', 3: 'SPR',
        4: 'SPR', 5: 'SPR', 6: 'SUM',
        7: 'SUM', 8: 'SUM', 9: 'AUT',
        10: 'AUT', 11: 'AUT', 12: 'WIN'
    }
    res = data.groupby(lambda x: seasons.get(x.month)).mean()
    return res


dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
dtypes = {
    'id': np.int32,
    'num_particles': np.int32,
    'temperature': np.int8,
    'atm_pressure': np.float32,
    'humidity': np.float32,
    'wind_direction': np.int8,
    'wind_speed': np.int8
}
cells_csv_path = 'cells_samples_csv'
pattern = re.compile("[0-9]+.csv")
for f in tqdm(os.listdir(cells_csv_path)):
    if pattern.match(f):
        #print(f)
        # read csv file for this cell
        csv_path = os.path.join(cells_csv_path, f)
        df = pd.read_csv(
            csv_path,
            na_values=['Nan'],
            parse_dates=['time'],
            date_parser=dateparse,
            dtype=dtypes
        ).set_index('time')
        df = df.drop('id', axis=1)
        group_daily(df).to_csv(csv_path[:-4] + '_daily.csv',
                               index_label='time')
        group_weekly(df).to_csv(csv_path[:-4] + '_weekly.csv',
                                index_label='time')
        group_monthly(df).to_csv(csv_path[:-4] + '_monthly.csv',
                                 index_label='time')
        group_seasonally(df).to_csv(csv_path[:-4] + '_seasonally.csv',
                                    index_label='time')
