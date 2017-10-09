from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Create some random data, I took this piece from here:
# http://matplotlib.org/mpl_examples/mplot3d/scatter3d_demo.py

target_names = ('EGY', 'GLF', 'LAV', 'MSA', 'NOR')
colors = ('black', 'turquoise', 'darkorange', 'green', 'blue')
marker = (2, 3, 4, 5, 6)

lw = 2

# def randrange(n, vmin, vmax):
#     return (vmax - vmin) * np.random.rand(n) + vmin
# n = 100
# xx = randrange(n, 23, 32)
# yy = randrange(n, 0, 100)
# zz = randrange(n, -50, -25)

X = np.loadtxt('scripts/feats')
y = np.loadtxt('scripts/labels')

# Create a figure and a 3D Axes
fig = plt.figure()
ax = Axes3D(fig)

#LDA
lda = LinearDiscriminantAnalysis(n_components=3)
X_r2 = lda.fit(X, y).transform(X)
# Create an init function and the animate functions.
# Both are explained in the tutorial. Since we are changing
# the the elevation and azimuth and no objects are really
# changed on the plot we don't have to return anything from
# the init and animate function. (return value is explained
# in the tutorial.
def init():
    ax.scatter(X_r2[y == 1, 0], X_r2[y == 1, 1], X_r2[y == 1, 2], label='Egyptian', alpha=0.6,
               c='r', marker='o', s=5, depthshade=False)
    ax.scatter(X_r2[y == 2, 0], X_r2[y == 2, 1], X_r2[y == 2, 2], label='Gulf', alpha=0.6,
               c='b', marker='^', s=5, depthshade=False)
    ax.scatter(X_r2[y == 3, 0], X_r2[y == 3, 1], X_r2[y == 3, 2], label='Levantine', alpha=0.6,
               c='y', marker='x', s=5, depthshade=False)
    ax.scatter(X_r2[y == 4, 0], X_r2[y == 4, 1], X_r2[y == 4, 2], label='MSA', alpha=0.6,
               c='c', marker='P', s=5, depthshade=False)
    ax.scatter(X_r2[y == 5, 0], X_r2[y == 5, 1], X_r2[y == 5, 2], label='Moroccan', alpha=0.6,
               c='g', marker='D', s=5, depthshade=False)
    # ax.scatter(xx, yy, zz, marker='o', s=20, c="goldenrod", alpha=0.6)
    return fig,

def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,

# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)
# Save
anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])