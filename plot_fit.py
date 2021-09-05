#!/usr/bin/env python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col
from matplotlib import ticker, cm
import pickle
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


A = np.load("col.npy")
fig, ax = plt.subplots(figsize=(7, 6))
#im = ax.imshow(A, cmap='Blues', vmin=0.0)
im = ax.imshow(A, cmap='Blues')

ax.set_ylabel("$u0$")
ax.set_xlabel(r"$u1$")

plt.show()
#plt.savefig("fun.png", dpi=200)
