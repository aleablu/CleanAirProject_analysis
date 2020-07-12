import pandas as pd
import numpy as np
import time

from pymongo import MongoClient



client = MongoClient('mongodb://127.0.0.1', port=27017)

db = client.cleanair

n_cells = 28 * 26

start_time = time.time()

for i in range(n_cells):
	cell = db.cells.find_one({'id': i})
	#print(cell['location'])
	query = {
		'location': {
			'$geoWithin': {
				'$geometry': cell['location']
 			}
		}
	}
	# count data in cell
	n_data = db.data.count_documents(query)
	
	if n_data >= 50:
		print('id = ' + str(i) + ', n_data = ' + str(n_data))
		s = time.time()
		# create dataframe from cell's data
		df = pd.DataFrame(list(db.data.find(query))).set_index('time')
		# drop not needed data

		df = df.drop(['_id', 'location', 'hdop', 'diam_particles', 'ldsa'], axis=1)
		df = df.sort_index()
		fname = 'cells_samples_csv/' + str(i) + '.csv'
		# save to csv
		df.to_csv(fname, index=False)
		# save reference to csv in mongo
		db.cells.update_one({'id': i}, {'$set': {'labeled': True, 'samples_csv_path': fname}})
		print('samples loaded, duration ' + str(time.time() - s) + ' seconds')
	else:
		# no data in cell, labeled = False and no csv path	
		db.cells.update_one({'id': i}, {'$set': {'labeled': False, 'samples_csv_path': ''}})

print('end, duration ' + str(time.time() - start_time) + ' seconds')