__author__ = 'efiathieniti'

import numpy as np
from pylab import *
from scipy.ndimage import measurements



def calculate_sizes(lattice):
    #
    # Calculate degree distribution for the final configuration
    # 
    plt.figure()
    lns = plt.plot()
    plt.title("size distribution")


    final_lattice = lattice[:,:,-1]
    final_lattice[nonzero(final_lattice)]=1
    lw, num = measurements.label(final_lattice)
    area = measurements.sum(final_lattice, lw, index=arange(lw.max() + 1))
    plt.hist(area, bins=100)

    plt.savefig('lattice_plots/hist.png')

lattice = np.load('lattice.npy')
calculate_sizes(lattice)