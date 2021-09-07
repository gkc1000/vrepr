from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv("FINALdata_coarse_partial01201")
fig = plt.figure()
ax = fig.gca(projection='3d')
#Z = df['gap']
#print(min(Z))
#Z[abs(Z)>4]=np.NaN
ax.plot_trisurf(df["n"], df["angle"], df["wt"])
#ax.set_zlim((-4,4))
plt.show()
