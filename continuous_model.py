import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import norm

lip_no = 100
dimensions = 20
T = 0.1
N = 100

dxys = norm.rvs(size=(2, lip_no, N))

lip_heads = np.zeros((2, lip_no, N))
lip_tails = np.zeros((2, lip_no, N))


# maybe make sure the same rand. no is not chosen twise
for i in range(lip_no):
    lip_heads[0, i, 0] = np.random.rand() * dimensions 
    lip_heads[1, i, 0] = np.random.rand() * dimensions
    rand = np.random.rand()
    lip_tails[0, i, 0] = lip_heads[0, i, 0] + 0.4*np.sin(rand * 2 * np.pi)
    lip_tails[1, i, 0] = lip_heads[1, i, 0] - 0.4*np.cos(rand * 2 * np.pi)

for n in range(1, N):
	for i in range(lip_no):
		lip_heads[:, i, n] = np.clip(lip_heads[:, i, n-1] + dxys[:, i, n], 1, dimensions-1)
		rand = np.random.rand()
		lip_tails[0, i, n] = lip_heads[0, i, n] + 0.4*np.sin(rand * 2 * np.pi)
		lip_tails[1, i, n] = lip_heads[1, i, n] - 0.4*np.cos(rand * 2 * np.pi)

for i in range(1,N):
    plt.figure(i-1, figsize=(6,6))
    axes = plt.gca()
    axes.set_xlim([0,dimensions])
    axes.set_ylim([0,dimensions])
    plt.plot([lip_tails[0,:,i], lip_heads[0,:,i]], [lip_tails[1,:,i], lip_heads[1,:,i]], 'k-')

    for j in range(lip_no):
        plt.plot(lip_tails[0,j,i], lip_tails[1,j,i], 'r.')
    plt.savefig('plots/continuous_%03d.png'%(i))
    #plt.show()
    plt.close(i-1)