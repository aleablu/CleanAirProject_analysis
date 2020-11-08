import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm
import time
import numpy as np

CELLS_DIMENSION = 250

def add_neighbours(path):
    df = pd.read_csv(path)
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        nn = df_neighbours.loc[df_neighbours['cell_id'] == row['cell_id']]
        for i in range(5):
            df.loc[idx, f"neigh_{i+1}_id"] = int(nn[f"neigh_{i+1}"].values[0])
            df.loc[idx, f"dist_{i+1}"] = nn[f"dist_{i+1}"].values[0]
    for i in range(5):
        df[f"neigh_{i+1}_id"] = df[f"neigh_{i+1}_id"].astype(int)
    return df


df_neighbours = pd.read_csv(f"../data/neighbouring_cells_{CELLS_DIMENSION}m.csv")

df_d = add_neighbours(f"../data/{CELLS_DIMENSION}m_merged_daily.csv")
df_w = add_neighbours(f"../data/{CELLS_DIMENSION}m_merged_weekly.csv")
df_m = add_neighbours(f"../data/{CELLS_DIMENSION}m_merged_monthly.csv")
df_s = add_neighbours(f"../data/{CELLS_DIMENSION}m_merged_seasonally.csv")


df_d.to_csv(f"../data/{CELLS_DIMENSION}m_merged_daily_added_neighbours.csv")
df_w.to_csv(f"../data/{CELLS_DIMENSION}m_merged_weekly_added_neighbours.csv")
df_m.to_csv(f"../data/{CELLS_DIMENSION}m_merged_monthly_added_neighbours.csv")
df_s.to_csv(f"../data/{CELLS_DIMENSION}m_merged_seasonally_added_neighbours.csv")