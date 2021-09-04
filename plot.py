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
im = ax.imshow(A, cmap='Blues', vmin=0.0)

ax.set_ylabel("$n$")
ax.set_xlabel(r"$\theta$")

plt.show()
#plt.savefig("fun.png", dpi=200)
