import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm
import time
import numpy as np


CELLS_DIMENSION = 250

client = MongoClient('mongodb://127.0.0.1', port=27017)
db = client.cleanair
n_cells = 56 * 52
start_time = time.time()

df = pd.DataFrame(columns=['cell_id', 'neigh_1', 'dist_1', 'neigh_2', 'dist_2', 'neigh_3', 'dist_3', 'neigh_4', 'dist_4', 'neigh_5', 'dist_5'])

for i in tqdm(range(n_cells)):
    cell = db.cells.find_one({'id': i})
    query = {
        '$geoNear': {
            'near': {'type': 'Point', 'coordinates': cell['center']},
            'maxDistance': 2000,
            'distanceField': 'distance',
            'query': {'labeled': True}
            #'spherical': True
            }
    }
    res = db.cells.aggregate([query, {'$limit': 100}])
    row = {
        'cell_id': '',
        'neigh_1': '',
        'dist_1': '',
        'neigh_2': '',
        'dist_2': '',
        'neigh_3': '',
        'dist_3': '',
        'neigh_4': '',
        'dist_4': '',
        'neigh_5': '',
        'dist_5': '',
    }
    row['cell_id'] = i
    for order, r in enumerate(res):
        if order < 5:
            row[f"neigh_{order+1}"] = int(r['id'])
            row[f"dist_{order+1}"] = r['distance']
        else:
            break
    df = df.append(row, ignore_index=True)

df.to_csv(f"../data/neighbouring_cells_{CELLS_DIMENSION}m.csv", index=False)

print(df.info())