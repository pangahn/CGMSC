import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# file_path = '../result/K/BBCsport_544_2views_5clusters_K.csv'
# file_path = '../result/K/ORL_400_4views_40clusters_K.csv'
# file_path = '../result/K/COIL-20_1440_3views_20clusters_K.csv'
# file_path = '../result/K/UCI-digit_2000_6views_10clusters_K.csv'
# file_path = '../result/K/3Sources_169_3views_6clusters_K.csv'
file_path = '../result/K/MSRC-v1_210_5views_7clusters_K.csv'

df = pd.read_csv(file_path)
data = df.sort_values(by=['K'])
X = data['K']
ACC = data['Accuracy']
NMI = data['NMI']
F1 = data['F1']
ARI = data['ARI']


fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111)

plt.plot(X, ACC,
        marker='o', markersize=8, alpha=0.9,
        color='b',
        linestyle='-', linewidth=2,
        label='ACC')

plt.plot(X, NMI,
        color='r',
        marker='^', markersize=10, alpha=0.9,
        linestyle='-.', linewidth=2,
        label='NMI')

plt.plot(X, F1,
        color='g',
        marker='X', markersize=10, alpha=0.9,
        linestyle='--', linewidth=2,
        label='F1')

plt.plot(X, ARI,
        color='black',
        marker='s', markersize=10, alpha=0.9,
        linestyle=':', linewidth=2,
        label='ARI')


ax.set_ylabel('Clustering Metrics Values', fontsize=20)
ax.set_xlabel('$K$', fontsize=25)
ax.set_xticks(X)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.legend(fontsize=18, loc='upper right')#, bbox_to_anchor=(1, 0.9))
plt.grid(linestyle='-.')
plt.tight_layout()
plt.show()