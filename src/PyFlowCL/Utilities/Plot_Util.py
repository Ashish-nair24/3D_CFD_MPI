import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from matplotlib import cm


# ------------------------------------------------------
# Grid plotter
# ------------------------------------------------------
def plot_grid(grid,outDir):
    plt.pcolormesh(grid.X_cpu,grid.Y_cpu,np.ones_like(grid.X_cpu),
                   cmap=cm.Greys,ec='black',LineWidth=0.25)
    #plt.xlim([win_xmin,win_xmax])
    #plt.ylim([win_ymin,win_ymax])
    plt.axis('square')
    plt.savefig(outDir+'/grid.pdf')
    plt.close()
