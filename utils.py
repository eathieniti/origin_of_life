__author__ = 'efiathieniti'

import numpy as np
from pylab import *
from scipy.ndimage import measurements
import copy

def calculate_sizes_zero_for_empty(lattice_snapshot, fname, loglog_scale=None):
    #
    # Calculate degree distribution for the final configuration
    #


    #print(final_lattice)

    ls = lattice_snapshot.copy()
    if -1 in ls:
        ls[ls!=-1]=1
        ls[ls==-1]=0
    lw, num = measurements.label(ls)


    plt.figure()
    lns = plt.plot()
    plt.title("size distribution")
    area = measurements.sum(ls, lw, index=arange(lw.max() + 1))
    area = area[area>1]
    plt.hist(area, bins=100)
    average_object_size = np.mean(area)
    median_object_size = np.median(area)
    print("average object size = ", average_object_size)
    print("median object size = ", median_object_size)
    if loglog_scale:
            plt.xscale('log')
            plt.yscale('log')


    plt.savefig(fname)





def calculate_sizes(lattice_snapshot, fname, loglog=None):
    #
    # Calculate degree distribution for the final configuration
    #

    ls = lattice_snapshot.copy()
    ls[ls!=-1]=1
    ls[ls==-1]=0
    lw, num = measurements.label(ls)


    plt.figure()
    lns = plt.plot()
    plt.title("size distribution")
    area = measurements.sum(ls, lw, index=arange(lw.max() + 1))
    area = area[area>1]
    plt.hist(area, bins=100)
    average_object_size = np.mean(area)
    median_object_size = np.median(area)
    print("average object size = ", average_object_size)
    print("median object size = ", median_object_size)
    if loglog:
            plt.xscale('log')
            plt.yscale('log')


    plt.savefig(fname)

#lattice = np.load('lattice_100000_x_100_y_100.npy')

#calculate_sizes(lattice)
