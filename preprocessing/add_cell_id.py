import pandas as pd
import numpy as np


def gen_cells():
	# num_celle_lungo = 28 # 500
	# num_celle_corto = 26
	num_celle_lungo = 56 # 250
	num_celle_corto = 52
	# meters = 0.006372 # 500 metri
	meters = 0.003186 # 250 metri
	lungo = []
	# sommo, mi sto muovendo verso est
	for i in range(0, num_celle_lungo):
		lungo.append((47.437045, 8.438894 + ( meters * i)))

	corto = []
	# sottraggo, mi sto muovendo verso sud
	for i in range(0, num_celle_corto):
		corto.append((47.437045 + (-meters * i), 8.438894))

	m = []
	for i in range(num_celle_corto):
		m.append([])
		for j in range(num_celle_lungo):
			start = corto[i]
			point = (start[0], start[1] + ( meters * j))
			m[i].append(point)
	return m



def add_id(df, cells):
    df['lat'] = [0 for k in range(len(df))]
    df['lon'] = [0 for k in range(len(df))]
    cell_id = 0
    for i in range(len(cells)):
        for j in range(len(cells[i])):
            indexes = df[df['cell_id'] == cell_id].index
            n = len(indexes)
            if n > 0:
                df.loc[indexes, ['lat']] = [cells[i][j][0] for k in range(n)]
                df.loc[indexes, ['lon']] = [cells[i][j][1] for k in range(n)]
            cell_id += 1
    print(df)


df_d = pd.read_csv('data/250m_merged_daily.csv')
df_w = pd.read_csv('data/250m_merged_weekly.csv')
df_w['time'] = df_w['time'].astype('int32')
df_w['cell_id'] = df_w['cell_id'].astype('int32')
df_m = pd.read_csv('data/250m_merged_monthly.csv')
df_s = pd.read_csv('data/250m_merged_seasonally.csv')

mat = gen_cells()

add_id(df_d, mat)
add_id(df_w, mat)
add_id(df_m, mat)
add_id(df_s, mat)

df_d.to_csv('data/merged_daily_updated.csv', index=False)
print('saved')
df_w.to_csv('data/merged_weekly_updated.csv', index=False)
print('saved')
df_m.to_csv('data/merged_monthly_updated.csv', index=False)
print('saved')
df_s.to_csv('data/merged_seasonally_updated.csv', index=False)
print('saved')
