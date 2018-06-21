import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import norm
from scipy.optimize import minimize

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
    lip_tails[0, i, 0] = lip_heads[0, i, 0] + 0.4*np.cos(rand * 2 * np.pi)
    lip_tails[1, i, 0] = lip_heads[1, i, 0] + 0.4*np.sin(rand * 2 * np.pi)

def dist_constraint(pos1, x2, y2):
    return ((pos1[0]-x2)**2 + (pos1[1]-y2)**2)**0.5 - 0.4

# def jac_dist_constraint(pos1, x2, y2):
#     return ()
    
def distance(pos, others): 
    return np.sum(np.sum((others.T - pos)**2, axis=1)**0.5, axis=0)

a = np.zeros((lip_no, N))
for n in range(1, N):
	lip_heads[:, :, n] = np.clip(lip_heads[:, :, n-1] + dxys[:, :, n] * a[:, n-1], 1, dimensions-1)
	lip_tails[:, :, n] = np.clip(lip_tails[:, :, n-1] + dxys[:, :, n] * a[:, n-1], 1, dimensions-1)
	for i in range(lip_no):
		neighbours = lip_tails[:, ((lip_tails[0, :, n] - lip_tails[0, i, n])**2 + (lip_tails[1, :, n] - lip_tails[1, i, n])**2)**0.5 < 5, n]
 		res = minimize(distance, lip_tails[:, i, n], args=(neighbours),
 		               constraints={"type": "eq", "fun": dist_constraint, "args": lip_heads[:, i, n]})
 		lip_tails[:, i, n] = res.x
 		a[i, n] = res.fun / (neighbours.shape[1]*10)
 	print n
        

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
