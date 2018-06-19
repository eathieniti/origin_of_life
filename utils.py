__author__ = 'efiathieniti'

import numpy as np
from pylab import *
from scipy.ndimage import measurements



def calculate_sizes(lattice, fname):
    #
    # Calculate degree distribution for the final configuration
    #

    final_lattice = lattice
    #print(final_lattice)

    final_lattice[final_lattice!=-1]=1
    final_lattice[final_lattice==-1]=0
    lw, num = measurements.label(final_lattice)


    plt.figure()
    lns = plt.plot()
    plt.title("size distribution")
    area = measurements.sum(final_lattice, lw, index=arange(lw.max() + 1))
    plt.hist(area, bins=100)
    average_object_size = np.mean(area)
    median_object_size = np.median(area)
    print("average object size = ", average_object_size)
    print("median object size = ", median_object_size)

    plt.savefig('%s.png'%fname)

#lattice = np.load('lattice_100000_x_100_y_100.npy')

#calculate_sizes(lattice)