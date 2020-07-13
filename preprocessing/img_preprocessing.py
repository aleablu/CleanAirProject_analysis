import os
import cv2
from tqdm import tqdm


out_size = 128

cells_images_dir_path = 'data/500m_cells_images'

imgs_out_path = 'data/500m_cells_images/resized_128'

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

