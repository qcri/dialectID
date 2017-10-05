# -*- coding: utf-8 -*-

import sys
import numpy as np

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

import matplotlib
# matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline

target_names = ('EGY', 'GLF', 'LAV', 'MSA', 'NOR')
colors = ('black', 'turquoise', 'darkorange', 'green', 'blue')
marker = (2, 3, 4, 5, 6)

lw = 2

###
#PCA and LDA for given features
###
X = np.loadtxt('scripts/feats')
y = np.loadtxt('scripts/labels')

#LDA
lda = LinearDiscriminantAnalysis(n_components=3)
X_r2 = lda.fit(X, y).transform(X)


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_r2[y == 1, 0], X_r2[y == 1, 1], X_r2[y == 1, 2], label='Egyptian',
           c='r', marker='o', s=5, depthshade=False)
ax.scatter(X_r2[y == 2, 0], X_r2[y == 2, 1], X_r2[y == 2, 2], label='Gulf',
           c='b', marker='^', s=5, depthshade=False)
ax.scatter(X_r2[y == 3, 0], X_r2[y == 3, 1], X_r2[y == 3, 2], label='Levantine',
           c='y', marker='x', s=5, depthshade=False)
ax.scatter(X_r2[y == 4, 0], X_r2[y == 4, 1], X_r2[y == 4, 2], label='MSA',
           c='c', marker='P', s=5, depthshade=False)
ax.scatter(X_r2[y == 5, 0], X_r2[y == 5, 1], X_r2[y == 5, 2], label='Moroccan',
           c='g', marker='D', s=5, depthshade=False)

# Make legend, set axes limits and labels
ax.legend()

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.autoscale(tight=True)
plt.show()