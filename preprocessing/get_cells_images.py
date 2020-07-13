import requests
import shutil
import math
from tqdm import tqdm


def get_tile(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return zoom, xtile, ytile


def gen_cells():
	num_celle_lungo = 28 # 144
	num_celle_corto = 26 # 130

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


mat = gen_cells()
k = 0
for i in tqdm(range(len(mat))):
    for j in range(len(mat[i])):
        zoom, x, y = get_tile(mat[i][j][0], mat[i][j][1], 15)
        r = requests.get('https://tiles.wmflabs.org/osm-no-labels/' + str(zoom)
                         + '/' + str(x) + '/' + str(y) + '.png', stream=True)
        if r.status_code == 200:
            with open('data/500m_cells_images/' + str(k) + '.png', 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
                k += 1
