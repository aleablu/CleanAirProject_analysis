from pymongo import MongoClient, GEOSPHERE


from datetime import datetime

import pandas as pd
import numpy as np
import math

import requests
import shutil


def get_tile(lat_deg, lon_deg, zoom):
	lat_rad = math.radians(lat_deg)
	n = 2.0 ** zoom
	xtile = int((lon_deg + 180.0) / 360.0 * n)
	ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
	return zoom, xtile, ytile

def gen_cells():
	num_celle_lungo = 28
	num_celle_corto = 26

	five_hundred_meters = 0.006372
	lungo = []
	# sommo, mi sto muovendo verso est
	for i in range(0, num_celle_lungo):
		lungo.append((47.437045, 8.438894 + ( five_hundred_meters * i)))

	corto = []
	# sottraggo, mi sto muovendo verso sud
	for i in range(0, num_celle_corto):
		corto.append((47.437045 + (-five_hundred_meters * i), 8.438894))

	m = []
	for i in range(num_celle_corto):
		m.append([])
		for j in range(num_celle_lungo):
			start = corto[i]
			point = (start[0], start[1] + ( five_hundred_meters * j))
			m[i].append(point)
	return m

# loads cells on mongodb instance
def load_cells_mongo(db):
	mat = gen_cells()  # obtain matrix of cell's centers
	db.cells.create_index([('location', GEOSPHERE)])
	id = 0
	cells = []
	for i in range(len(mat)):
		for j in range(len(mat[i])):
			# calc cell corner coord
			fifty_meters = 0.001309
			five_hundred_meters = 0.006372
			nw = [mat[i][j][0] + five_hundred_meters, mat[i][j][1] - five_hundred_meters]
			se = [mat[i][j][0] - five_hundred_meters, mat[i][j][1] + five_hundred_meters]
			ne = [nw[0], se[1]]
			sw = [se[0], nw[1]]
			coord = [nw, ne, se, sw, nw]
			# create cell dict
			cell = {
				'id': id,
				'location': {
					'type': 'Polygon',
					'coordinates': [coord]
				},
				'center': mat[i][j],
				'image_path': 'cells_images/'+str(id)+'.png',
				'air_quality': 0
			}
			cells.append(cell)
			id += 1
	db.cells.insert_many(cells)
	print('done, cells loaded to mongodb')


def load_dataset_mongo(db):
	dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
	dtypes = {
		'lat': np.float32,
		'lon': np.float32,
		'hdop': np.float32,
		'tram_id': np.int8,
		'num_particles': np.int32,
		'diam_particles': np.float32,
		'ldsa': np.float32,
		'temperature': np.int8,
		'atm_pressure': np.float32,
		'humidity': np.float32,
		'wind_direction': np.int8,
		'wind_speed': np.int8
	}
	
	db.data.create_index([('location', GEOSPHERE)])
	i = 1
	for data in pd.read_csv('../backups/whole_data_dropped_nans.csv', na_values=['Nan'], parse_dates=['time'], date_parser=dateparse, dtype=dtypes, chunksize=1000000):
		batch = []
		for index, row in data.iterrows():
			batch.append({
				'id': index,
				'time': row['time'],
				'location': {
					'type': 'Point',
					'coordinates': [row['lat'], row['lon']]
				},
				'hdop': row['hdop'],
				'num_particles': row['num_particles'],
				'diam_particles': row['diam_particles'],
				'ldsa': row['ldsa'],
				'temperature': row['temperature'],
				'atm_pressure': row['atm_pressure'],
				'humidity': row['humidity'],
				'wind_direction': row['wind_direction'],
				'wind_speed': row['wind_speed']
			})
		db.data.insert_many(batch)
		batch = []
		print('end of chunk ' + str(i))
		i += 1


client = MongoClient('mongodb://127.0.0.1', port=27017)  # connect to mongo container

db_mongo = client.cleanair  # create new db object called 'cleanair'

load_cells_mongo(db_mongo)  # load cells on mongodb

load_dataset_mongo(db_mongo)


# zoom, x, y = get_tile(47.437045, 8.438894, 16)

# r = requests.get('https://tiles.wmflabs.org/osm-no-labels/' + str(zoom) + '/' + str(x) + '/' + str(22933) + '.png', stream=True)
# if r.status_code == 200:
# 	with open('image.png' , 'wb') as f:
# 		r.raw.decode_content=True
# 		shutil.copyfileobj(r.raw, f)
