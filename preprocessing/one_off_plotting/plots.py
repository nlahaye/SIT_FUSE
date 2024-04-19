

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


time1 = [[70, 1445, 1515, 1375], [1445, 70, 1442,  2873, 1431], [34, 1623, 3081, 1589, 3047, 1458]]

cent_dist1 = [[115.8, 105.8, 158.8, 26.7], [46.9, 54.2, 32.2, 10.4, 35.4], [389.4, 32.2, 8.4, 357.2, 390.5, 34.6]]

contour_diff1 = [[3.3, 576, 3.4, 133.2], [1.7, 1.4, 1, 0.2, 0.6], [0.8, 1, 0.2, 0.2, 0.5, 0.6]]
 
swd1 = [[143.1, 147.1, 163.5, 140.4], [146.2, 113.8, 1228, 1754.6, 895.9], [1211.6, 767.9, 1974.3, 823.7, 604.6, 1243.4]]

contour_diff = []
swd = []
time = []
cent_dist = []
for i in range(len(contour_diff1)):
    scl = MinMaxScaler()
    contour_diff.extend(scl.fit_transform(np.expand_dims(contour_diff1[i], axis=1)).flatten())

    scl = MinMaxScaler()
    swd.extend(scl.fit_transform(np.expand_dims(swd1[i], axis=1)).flatten())

    scl = MinMaxScaler()
    time.extend(scl.fit_transform(np.expand_dims(time1[i], axis=1)).flatten())

    scl = MinMaxScaler()
    cent_dist.extend(scl.fit_transform(np.expand_dims(cent_dist1[i], axis=1)).flatten())


print(swd)
print(contour_diff)

data = {
"Time": time,
"Distance": cent_dist,
"Shape Diff": contour_diff,
"Clust SWD": swd}


df = pd.DataFrame(data)

cov_matrix = pd.DataFrame.cov(df)
sn.heatmap(cov_matrix, annot=True, fmt='g')
plt.show()
plt.savefig("Smoke_Contour_Heatmap.png", dpi=400)
plt.clf()



corr_matrix = df.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()
plt.savefig("Smoke_Contour_Heatmap_2.png", dpi=400)



sn.pairplot(df)
plt.show()
plt.savefig("Pairplot.png", dpi=400)



for i in range(0,4):
    #print(swd[i], contour_diff[i], time[i], cent_dist[i], time[i]*cent_dist[i])
    print(i, ((swd[i] + contour_diff[i])/((time[i]+1e-8)*(cent_dist[i]+1e-8))))


for i in range(4,8):
    #print(swd[i], contour_diff[i], time[i], cent_dist[i], time[i]*cent_dist[i])
    print(i, ((swd[i] + contour_diff[i])/((time[i]+1e-8)*(cent_dist[i]+1e-8))))


for i in range(8,15):
    #print(swd[i], contour_diff[i], time[i], cent_dist[i], time[i]*cent_dist[i])
    print(i, ((swd[i] + contour_diff[i])/((time[i]+1e-8)*(cent_dist[i]+1e-8))))


