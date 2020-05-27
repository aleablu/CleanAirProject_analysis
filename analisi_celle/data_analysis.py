import pandas as pd 
import numpy as np 
import os
import sys
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import matplotlib
import cv2
import seaborn as sns

DATA_DIR = "cells_samples_csv/"

if len(sys.argv) > 1 and sys.argv[1] == '-u':
	UPDATE = True
else:
	UPDATE = False

no_data_cell = {
			'count': 			0,
			'num_particles': 	0,
			'temperature': 		0,
			'atm_pressure': 	0,
			'humidity': 		0,
			'wind_direction': 	0,
			'wind_speed': 		0
			}

if UPDATE:

	files = os.listdir(DATA_DIR)
	df = pd.DataFrame()

	for i in tqdm(range(130 * 144)):
		csv = DATA_DIR + str(i) + '.csv'
		# se file della cella esiste la cella è informata
		if os.path.exists(csv):
			curr = pd.read_csv(csv)
			curr = curr.drop(["time","id"], axis=1)

			new = {}
		
			new["id_cell"] = i
			new["count"] = curr.shape[0]

			for col in curr.columns:

				new[col] = curr[col].mean()

			df = df.append(new, ignore_index=True)
		else:
			# se non esiste, carico una cella con 0 misurazioni e NaN nel resto
			no_data_cell['id_cell'] = i
			df = df.append(no_data_cell, ignore_index=True)

	pickle.dump(df, open("data_mean.pkl", "wb"))
	df.to_csv('data_mean.csv', index=False)
else:
	df = pickle.load(open("data_mean.pkl", "rb"))



def plot(df):

	plt.plot(df["count"])
	plt.xticks(range(1, len(df["id_cell"]), 100), df["id_cell"], rotation=90)
	plt.grid(True)
	plt.title("Numero misurazioni per cella")
	plt.show()

	plt.plot(df["temperature"].rolling(100).mean())
	plt.xticks(range(1, len(df["id_cell"]), 100), df["id_cell"], rotation=90)
	plt.grid(True)
	plt.title("temperatura media per cella")
	plt.show()

	plt.plot(df["num_particles"])
	plt.plot([df["num_particles"].mean() for i in range(len(df["num_particles"]))])
	plt.plot([np.std(df["num_particles"]) for i in range(len(df["num_particles"]))])
	plt.xticks(range(1, len(df["id_cell"]), 100), df["id_cell"], rotation=90)
	plt.grid(True)
	plt.legend(["numero particelle", "media", "std"])
	plt.title("Particelle per cella")
	plt.show()

	plt.pcolor(df)
	plt.xticks(np.arange(1, len(df.columns), 1), df.columns)
	plt.title("Heatmap DF")
	plt.show()

def heat_map(df, feature, title, alpha):
	
	data = np.array(df[feature])
	
	heat_map = data.reshape(-1, 144)
	# read image
	map_img = plt.imread('map-561x503.png')

	hmax = sns.heatmap(heat_map,
			cmap = matplotlib.cm.winter,
			alpha = alpha, # whole heatmap is translucent
			zorder = 2,
        )
	hmax.imshow(map_img,
          aspect = hmax.get_aspect(),
          extent = hmax.get_xlim() + hmax.get_ylim(),
          zorder = 1) #put the map under the heatmap

	plt.title(title)
	plt.savefig(f"imgs/{feature}_heatmap.png")
	plt.show()
df = df.set_index(["id_cell"])
df = df.sort_index()

heat_map(df, 'count', 'Numero di rilevazioni per cella', 0.7)

heat_map(df, 'num_particles', 'Numero medio di particelle per cella', 0.5)

