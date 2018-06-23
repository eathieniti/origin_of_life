import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt

from multiprocessing import Process

from scipy.stats import norm
from scipy.optimize import minimize

lip_no = 200
dimensions = 20
T = 0.1
N = 300

dir_step_size = 0.1
noise_scale = 0.1
infl_repulsion_heads = 0.2 
detection_radius = 5.
lipid_length = 0.4
lip_len_scale = 0.
min_tail_dist = 0.1

noise = norm.rvs(size=(2, lip_no, N)) * noise_scale

lip_heads = np.zeros((2, lip_no, N))
lip_tails = np.zeros((2, lip_no, N))
lip_lengths = norm.rvs(size=lip_no) * lip_len_scale + lipid_length


# maybe make sure the same rand. no is not chosen twise
for i in range(lip_no):
    lip_heads[0, i, 0] = np.random.rand() * dimensions 
    lip_heads[1, i, 0] = np.random.rand() * dimensions
    rand = np.random.rand()
    lip_tails[0, i, 0] = lip_heads[0, i, 0] + lip_lengths[i]*np.cos(rand * 2 * np.pi)
    lip_tails[1, i, 0] = lip_heads[1, i, 0] + lip_lengths[i]*np.sin(rand * 2 * np.pi)

def dist_constraint(pos, lip_len):
    """
    This is the constraint to keep the distance between the head and the tail constant.
    """
    return ((pos[0]-pos[2])**2 + (pos[1]-pos[3])**2)**0.5 - lip_len

def small_step_constraint(pos, x_old, y_old):
    """
    This is the constraint for not making too large steps.
    x_old and y_old are the old positions of the mass points
    slowing down the movement because of friction with water
    max step size 0.1
    """
    return -(((x_old - (pos[0]+pos[2])/2.)**2 + (y_old - (pos[1]+pos[3])/2.)**2)**0.5 - dir_step_size)

def tails_constraint(pos, tails):
    return min(np.sum((tails.T - pos[:2])**2, axis=1)**0.5) - min_tail_dist

# def jac_dist_constraint(pos1, x2, y2):
#     return ()
    
def distance(pos, tails, heads):
    """
    pos: [tail.x, tail.y, head.x, head.y]
    """ 
    return (np.sum(np.sum((tails.T - pos[:2])**2, axis=1)**0.5, axis=0) - 
            infl_repulsion_heads * np.sum(np.sum((heads.T - pos[2:])**2, axis=1)**0.5, axis=0))


def visualization(i):
    plt.figure(i, figsize=(6,6))
    axes = plt.gca()
    axes.set_xlim([0,dimensions])
    axes.set_ylim([0,dimensions])
    plt.plot([lip_tails[0,:,i], lip_heads[0,:,i]], [lip_tails[1,:,i], lip_heads[1,:,i]], 'k-')
    for j in range(lip_no):
        plt.plot(lip_tails[0,j,i], lip_tails[1,j,i], 'r.')
    plt.savefig('plots/continuous_%03d.png'%(i))
    #plt.show()
    plt.close(i)

a = np.zeros((lip_no, N))
for n in range(1, N):
    lip_heads[:, :, n] = lip_heads[:, :, n-1]
    lip_tails[:, :, n] = lip_tails[:, :, n-1]
    for i in np.random.permutation(lip_no):
        neighbours = np.array(((lip_tails[0, :, n] - lip_tails[0, i, n])**2 + (lip_tails[1, :, n] - lip_tails[1, i, n])**2)**0.5 < detection_radius)
        if sum(neighbours) == 0:
            continue
        masspoint = ((lip_heads[0, i, n] + lip_tails[0, i, n])/2., (lip_heads[1, i, n] + lip_tails[1, i, n])/2.)
        res = minimize(distance, np.concatenate((lip_tails[:, i, n], lip_heads[:, i, n])), 
                       args=(lip_tails[:, neighbours, n], lip_heads[:, neighbours, n]),
                       constraints=({"type": "eq", "fun": dist_constraint, "args": (lip_lengths[i],)}, 
                                    {"type": "ineq", "fun": small_step_constraint, "args": masspoint},
                                    {"type": "ineq", "fun": tails_constraint, "args": (lip_tails[:, neighbours, n],)}))
        new_pos = res.x
        lip_tails[:, i, n] = np.clip(new_pos[:2] + noise[:, i, n], 1, dimensions-1)
        lip_heads[:, i, n] = np.clip(new_pos[2:] + noise[:, i, n], 1, dimensions-1)
    print n
    p = Process(target=visualization, args=(n,))
    p.start()
    p.join()
    #visualization(n)
        

# for i in range(N):
#     visualization(i)
