import matplotlib
import matplotlib.pyplot as plt
import numpy as np

width = 1

data = np.array([[0.0858, 0.887,0.1247, 0.4925, 13.4313, 3.8403],
                [0.0681, 0.3119,  0.1113, 0.3577, 11.9945, 3.4744],
                [0.1015, 0.7464,  0.1473, 0.5398, 39.0372, 9.0963],
                [0.1220, 0.8411,  0.2997, 0.7647, 116.981, 23.4762],
                [0.1216, 0.8116,  0.2884, 0.7102, 115.6267, 23.2952],
                [5.5885, 5.3348,  1.2615, 2.0406, 49.2147, 28.3464],
                [1.5469, 4.8487,  1.5895, 2.1318, 143.362, 58.28]])

datasets = ['3Sources', 'BBCSport','MSRCv1','ORL','UCI digits','COIL-20']
Algs = ['Co-reg P','Co-reg C','AMGL','PMLRSSC','CMLRSSC','MSC_IAS','Ours']
group_count = len(datasets)
idx_start = 4.5 * width
idx = np.arange(idx_start, 7*group_count + group_count, 8)

fig, ax = plt.subplots()

rects1 = ax.bar(idx - 3*width, data[0], width, label=Algs[0])
rects2 = ax.bar(idx - 2*width, data[1], width, label=Algs[1])
rects3 = ax.bar(idx - 1*width, data[2], width, label=Algs[2])
rects4 = ax.bar(idx - 0*width, data[3], width, label=Algs[3])
rects5 = ax.bar(idx + 1*width, data[4], width, label=Algs[4])
rects6 = ax.bar(idx + 2*width, data[5], width, label=Algs[5])
rects7 = ax.bar(idx + 3*width, data[6], width, label=Algs[6])

ax.set_xticks(idx)
ax.set_xticklabels(datasets, fontsize=20)
ax.set_ylabel('Time/s', fontsize=20)
ax.yaxis.set_tick_params(labelsize=18)
ax.legend(fontsize=15)
fig.tight_layout()
plt.grid(True, linestyle='-.')
plt.show()