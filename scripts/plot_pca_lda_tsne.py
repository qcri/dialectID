

# This code test with  conda 4.3.21

# Example for dimensionality reduction to 4 components for the five dialects...
# unsupervised; PCA and t-SNE algorithms & supervisied LDA
#
#TODO: Maybe 3D to visualize features better
#use the new features in training...

print(__doc__)


import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import numpy as np
import sys

target_names = ['EGY','GLF','LAV','MSA','NOR']
colors = ['navy', 'turquoise', 'darkorange', 'green', 'blue']
lw = 2


###
#PCA and LDA for given features
###
X = np.loadtxt('feats')    
y = np.loadtxt('labels')    



#PCA
pca = PCA(n_components=4)
X_r = pca.fit(X).transform(X)
#np.savetxt('type.pca', X_r)
fig = plt.figure()
plt.clf()
for color, i, target_name in zip(colors, [0, 1, 2, 3, 4], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of dialect dataset')
#plt.show()
fig.savefig("pca.png", bbox_inches='tight')

# Percentage of variance explained for each components
#print('explained variance ratio (first two components): %s'
#      % str(pca.explained_variance_ratio_))

#LDA
lda = LinearDiscriminantAnalysis(n_components=4)
X_r2 = lda.fit(X, y).transform(X)
#np.savetxt('type.lda', X_r2)
plt.clf()
for color, i, target_name in zip(colors, [0, 1, 2, 3, 4], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of dialect dataset')
# Percentage of variance explained for each components
fig.savefig("lda.png", bbox_inches='tight')


#TSNE
model = TSNE(perplexity=30,n_components=4,init='pca',n_iter=5000)
np.set_printoptions(suppress=True)
X_r3=model.fit_transform(X)
#np.savetxt('type.tsne', X_r3)
plt.clf()
for color, i, target_name in zip(colors, [0, 1, 2, 3, 4], target_names):
    plt.scatter(X_r3[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('TSNE of dialect dataset')
fig.savefig("tsne.png", bbox_inches='tight')



