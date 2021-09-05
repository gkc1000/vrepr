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


A = np.load("u1_col.npy")[1:-1, 1:]
fig, ax = plt.subplots(figsize=(7, 6))
#im = ax.imshow(A, cmap='Blues', vmin=1e-12)
im = ax.imshow(A, cmap='coolwarm')

ax.set_ylabel("$n$")
ax.set_xlabel(r"$\theta$")

plt.show()
#plt.savefig("gap.png", dpi=200)
