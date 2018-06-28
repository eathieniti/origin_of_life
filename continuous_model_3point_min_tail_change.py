import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt
import os
from multiprocessing import Process

from scipy.stats import norm
from scipy.optimize import minimize

### 2-point model for the lipids
### Cost function includes:
###  1. tail1-tail1 attraction
###  2. tail2-tail2 attraction
### Constraints:
#    1. lipid length
#    2. small step
#    3. tail to tail > min_tail_dist
#    4. head to head > min_head_dist


# Params for bilayer!!
## original set
lip_no = 300
#dimensions = 20
dimensions = 20
T = 0.1
N = 999

dir_step_size = 0.3
noise_scale = 0.1
infl_repulsion_heads = 0.2
detection_radius = 1.
lipid_length = 0.65
lip_len_scale = 0.

# Important for bilayer
min_tail_dist = 0.3
min_head_dist = 0.4
min_tail_dist_max = 0.4

tails_n = 9 # Choose this so that N/tails_n is exactly divisible
min_tail_dists=np.linspace(0,min_tail_dist_max,tails_n)
min_tail_dists=np.repeat(min_tail_dists,N/tails_n,axis=0)

ex=6
out_dir = "plots_ll_%s_dr_%s_ss_%s_noise_%s_taildist_%s_ex_%s_N_%s_color"%(lipid_length,detection_radius, dir_step_size,noise_scale,min_tail_dist,ex, N )

try:
    os.mkdir(out_dir)
except:
    pass

noise = norm.rvs(size=(2, lip_no, N)) * noise_scale

lip_heads = np.zeros((2, lip_no, N))
lip_tails = np.zeros((2, lip_no, N))
lip_lengths = norm.rvs(size=lip_no) * lip_len_scale + lipid_length

# maybe make sure the same rand. no is not chosen twise
for i in range(lip_no):
    lip_heads[0, i, 0] = np.random.rand() * dimensions*0.9 + dimensions*0.05
    lip_heads[1, i, 0] = np.random.rand() * dimensions*0.9 + dimensions*0.05

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

def heads_constraint(pos, heads):
    return min(np.sum((heads.T - pos[2:])**2, axis=1)**0.5) - min_head_dist


def distance(pos, tails, heads, ex):
    """
    pos: [tail.x, tail.y, head.x, head.y]
    """
    sec_tails = heads.T + 0.2*(tails.T - heads.T)
    return (np.sum(np.sum((tails.T - pos[:2])**2, axis=1)**0.5, axis=0)
            + np.sum(np.sum((sec_tails - (pos[2:] + 0.2*(pos[:2] - pos[2:])))**2, axis=1)**0.5, axis=0)
            #- 0.5 * np.sum(np.sum((tails.T - pos[2:])**2, axis=1)**0.5, axis=0)
            )


def visualization(n, tails, heads, min_tail_dists):
    indices = (np.abs(np.sum((heads.T - tails.T)**2, axis=1)**0.5 - lip_lengths) < 0.1 )
    #print(indices)
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    ax[0].plot([tails[0, indices], heads[0, indices]], [tails[1, indices], heads[1, indices]], '-', color='darkgrey')
    ax[0].plot(heads[0, indices], heads[1,indices], 'b.')
    ax[0].set_xlim([0,dimensions])
    ax[0].set_ylim([0,dimensions])
    #ax[0].patch.set_facecolor('lightskyblue')
    #ax[0].patch.set_alpha(0.3)



    ax[1].plot(np.arange(len(min_tail_dists))[1:n],min_tail_dists[1:n])
    ax[1].set_xlim(1,len(min_tail_dists))
    ax[1].set_ylim(0,min_tail_dist_max)
    ax[1].set_xlabel("timestep")
    ax[1].set_ylabel("min_tail_dist")

    plt.savefig(out_dir + '/continuous_%03d.png'%(n))
    #plt.show()
    plt.close(n)

a = np.zeros((lip_no, N))
total_costs_N = np.zeros((lip_no,N))
for n in range(1, N):
    min_tail_dist=min_tail_dists[n]

    lip_heads[:, :, n] = lip_heads[:, :, n-1]
    lip_tails[:, :, n] = lip_tails[:, :, n-1]

    for i in np.random.permutation(lip_no):
        distances = ((lip_tails[0, :, n] - lip_tails[0, i, n])**2 + (lip_tails[1, :, n] - lip_tails[1, i, n])**2)**0.5
        neighbours = np.logical_and(0. < distances, distances < detection_radius)
        if sum(neighbours) == 0:
            new_pos = np.concatenate((lip_tails[:, i, n], lip_heads[:, i, n]))
            cost=0
        else:
            masspoint = ((lip_heads[0, i, n] + lip_tails[0, i, n])/2., (lip_heads[1, i, n] + lip_tails[1, i, n])/2.)
            res = minimize(distance, np.concatenate((lip_tails[:, i, n], lip_heads[:, i, n])), method="SLSQP",
                           args=(lip_tails[:, neighbours, n], lip_heads[:, neighbours, n], ex), options={"maxiter": 500},
                           constraints=({"type": "eq", "fun": dist_constraint, "args": (lip_lengths[i],)}, 
                                        {"type": "ineq", "fun": small_step_constraint, "args": masspoint},
                                        {"type": "ineq", "fun": tails_constraint, "args": (lip_tails[:, neighbours, n],)},
                                        {"type": "ineq", "fun": heads_constraint, "args": (lip_heads[:, neighbours, n],)}))
            new_pos = res.x
            cost = res.fun
        total_costs_N[i][n] = cost

        lip_tails[:, i, n] = np.clip(new_pos[:2] + noise[:, i, n], 1, dimensions-1)
        lip_heads[:, i, n] = np.clip(new_pos[2:] + noise[:, i, n], 1, dimensions-1)
    print(n)

    p = Process(target=visualization, args=(n, lip_tails[:, :, n], lip_heads[:, :, n], min_tail_dists))
    p.start()
    p.join()
    #visualization(n)
        


# Energy plot
total_costs=np.nansum(total_costs_N, axis=0)
#print(total_costs,total_costs.shape, N)
fig, ax1 = plt.subplots()
ax1.set_xlabel('time')
ax1.set_ylabel('H')
ax1.plot(np.arange(N), total_costs)
ax1.tick_params(axis='y')
plt.savefig(out_dir + "/energy_plot.png")


fig, ax1 = plt.subplots()
color = 'red'
ax1.set_xlabel('time')
ax1.set_ylabel('H')
ax1.plot(np.arange(N), total_costs)
ax1.tick_params(axis='y')


plt.savefig(out_dir + "/energy_plot.png")
