import numpy as np
import numpy.matlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# file_path = '../result/heatmap/3Sources_169_3views_6clusters_final.csv'
# file_path = '../result/heatmap/BBCsport_544_2views_5clusters_final.csv'
# file_path = '../result/heatmap/ORL_400_4views_40clusters_final.csv'
# file_path = '../result/heatmap/MSRC-v1_210_5views_7clusters_final.csv'
# file_path = '../result/heatmap/COIL-20_1440_3views_20clusters_final.csv'
file_path = '../result/heatmap/UCI-digit_2000_6views_10clusters_final.csv'
df = pd.read_csv(file_path)

fix_value = 79.4328
K_value = 10
fix_string = 'lambda3'
X_string = 'lambda1'
Y_string = 'lambda2'
Z_string = 'Accuracy'

data = df[(df[fix_string]==fix_value) & (df['K']==K_value)]
data = data.sort_values(by=[X_string, Y_string])

data = df[(df[fix_string]==fix_value) & (df['K']==K_value)]
data = data.sort_values(by=[X_string, Y_string])

dff = pd.pivot_table(data=data,  index=Y_string, values=Z_string, columns=X_string)

fig = plt.figure(figsize=(9,8))
ax = fig.add_subplot(111)

im = sns.heatmap(dff, annot=True, fmt=".2f", annot_kws={'size':18}, cmap='coolwarm', robust=True, cbar=False)

ax.invert_yaxis()
ax.set_xlabel(r'$\lambda_{}$'.format(X_string[-1]), fontsize=20)
ax.set_ylabel(r'$\lambda_{}$'.format(Y_string[-1]), fontsize=20)

ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)

plt.tight_layout()
plt.show()