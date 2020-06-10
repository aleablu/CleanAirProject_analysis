import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


out_size = 64

cells_images_dir_path = 'data/big_cells_images'

imgs_out_path = 'data/big_cells_images/resized_64'

for f in tqdm(os.listdir(cells_images_dir_path)):
	if 'png' in f:
		try:
			path = os.path.join(cells_images_dir_path, f)
                	# load image
			img = cv2.imread(path, cv2.IMREAD_COLOR)
			# resize image
			img = cv2.resize(img, (out_size, out_size))
			# save image
			cv2.imwrite(os.path.join(imgs_out_path, f), img)
			
		except Exception as e:
			print(e)
			pass

