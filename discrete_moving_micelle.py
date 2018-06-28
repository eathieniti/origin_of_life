"""
Model for the discrete lattice with continuous angle and micelle movement.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import random as r
from scipy.ndimage import measurements
import copy as cp


save_plots = True

timesteps = 100000
N = 40
N_lipids = 200


lattice = np.ones((N, N, timesteps)) * -1.

T = np.zeros(timesteps)
T_init = 1.
T[0] = T_init
T_decrease = 'linea'

density = float(N_lipids)/(N*N)
print("timesteps: ", timesteps, "N: ", N, "N_lipids: ",N_lipids)
print("density: ", density )

for i in range(N_lipids):
    lattice[np.random.randint(1, N-1), np.random.randint(1, N-1), 0] = np.random.random() * 2 * np.pi

def energy_f(lat):
    H_lat = (lat[1:N-1, 1:N-1] != -1.) * ((lat[0:N-2, 1:N-1] != -1.) * np.cos(lat[1:N-1, 1:N-1] + lat[0:N-2, 1:N-1]) + 
                                          (lat[2:N, 1:N-1] != -1.) * np.cos(lat[1:N-1, 1:N-1] + lat[2:N, 1:N-1]) + 
                                          (lat[1:N-1, 0:N-2] != -1.) * np.cos(lat[1:N-1, 1:N-1] + lat[1:N-1, 0:N-2] - np.pi) + 
                                          (lat[1:N-1, 2:N] != -1.) * np.cos(lat[1:N-1, 1:N-1] + lat[1:N-1, 2:N] - np.pi) +
                                          (lat[0:N-2, 0:N-2] != -1.) * np.cos(lat[1:N-1, 1:N-1] + lat[0:N-2, 0:N-2] - 0.5*np.pi) + 
                                          (lat[2:N, 2:N] != -1.) * np.cos(lat[1:N-1, 1:N-1] + lat[2:N, 2:N] - 0.5*np.pi) + 
                                          (lat[2:N, 0:N-2] != -1.) * np.cos(lat[1:N-1, 1:N-1] + lat[2:N, 0:N-2] - 1.5*np.pi) + 
                                          (lat[0:N-2, 2:N] != -1.) * np.cos(lat[1:N-1, 1:N-1] + lat[0:N-2, 2:N] - 1.5*np.pi) +
                                          (lat[0:N-2, 1:N-1] == -1.) * np.cos(lat[1:N-1, 1:N-1] - np.pi) + 
                                          (lat[2:N, 1:N-1] == -1.) * np.cos(lat[1:N-1, 1:N-1] - 2*np.pi) + 
                                          (lat[1:N-1, 0:N-2] == -1.) * np.cos(lat[1:N-1, 1:N-1] - 1.5*np.pi) + 
                                          (lat[1:N-1, 2:N] == -1.) * np.cos(lat[1:N-1, 1:N-1] - 0.5*np.pi) +
                                          (lat[0:N-2, 0:N-2] == -1.) * np.cos(lat[1:N-1, 1:N-1] - 1.25*np.pi) + 
                                          (lat[2:N, 2:N] == -1.) * np.cos(lat[1:N-1, 1:N-1] - 0.25*np.pi) + 
                                          (lat[2:N, 0:N-2] == -1.) * np.cos(lat[1:N-1, 1:N-1] - 1.75*np.pi) + 
                                          (lat[0:N-2, 2:N] == -1.) * np.cos(lat[1:N-1, 1:N-1] - 0.75*np.pi)
                                         )
    return np.sum(H_lat)


def move_chunk(lattic):
    # copy the lattice to alter the changed version
    final = cp.deepcopy(lattic)
    finall = cp.deepcopy(lattic)
    final_lattice = cp.deepcopy(lattic)
    for x in range(N):
        # change arrows value to 1 and empty spots to 0 for detecting clusters
        for y in range(N):
            if final_lattice[x,y]!=-1:
                final_lattice[x,y]=1
            if final_lattice[x,y]==-1:
                final_lattice[x,y]=0  
    # pick a random percolation cluster to move
    lw, num = measurements.label(final_lattice)
    # make arrays to save the indices of the cluster sites and choose cluster
    while True:
        arx = []
        ary =[]
        ran=r.randint(1,np.amax(lw))
        for x in range(40):
            for y in range(40):
                if lw[x,y]==ran:
                    arx.append(x)
                    ary.append(y)
        if len(arx)>1:
            break
    # move the cluster into a random neuman direction
    if len(arx)>1:
        chance = r.random()
        if chance<0.25 and chance >=0 and max(ary)<39:
            for x in range(len(arx)):
                finall[arx[x],ary[x]] =-1
            for x in range(len(arx)):
                finall[arx[x],ary[x]+1] =final[arx[x],ary[x]]
        elif chance<0.5 and chance>=0.25 and min(ary)>0:
            for x in range(len(arx)):
                finall[arx[x],ary[x]] =-1
            for x in range(len (arx)):
                finall[arx[x],ary[x]-1] =final[arx[x],ary[x]]     
        elif chance<0.75 and chance>=0.5 and min(arx)>0:
            for x in range(len(arx)):
                finall[arx[x],ary[x]] =-1
            for x in range(len(arx)):
                finall[arx[x]-1,ary[x]] =final[arx[x],ary[x]]           
        elif chance>=0.75 and chance<1 and max(arx)<39:
            for x in range(len(arx)):
                finall[arx[x],ary[x]] =-1
            for x in range(len(arx)):
                finall[arx[x]+1,ary[x]] =final[arx[x],ary[x]]
    return(finall)

energy = np.zeros(timesteps)
energy[0] = energy_f(lattice[:, :, 0])

for n in range(1, timesteps):
    if n%1000==0:
        print n,"sim"
    
    lattice[:, :, n] = lattice[:, :, n-1]
    while True:
        i, j = np.random.randint(1, N-1), np.random.randint(1, N-1)
        if lattice[i, j, n] != -1.:
            break
    if np.random.random() < 0.5:
        lattice[i, j, n] = np.random.random() * 2*np.pi
    else:
        while True:
            di, dj = np.random.randint(-1, 2), np.random.randint(-1, 2)
            if 0 < i+di < N-1 and 0 < j+dj < N-1:
                break
        lattice[i, j, n], lattice[i+di, j+dj, n] = lattice[i+di, j+dj, n-1], lattice[i, j, n-1]
    if r.random()<0.2:
        lattice[:, :, n]=move_chunk(lattice[:,:,n])
    energy[n] = energy_f(lattice[:, :, n])
    if T_decrease == 'linear':
        T[n] = T[n-1] - T_init/(timesteps)
    if np.random.random() >= np.e**((energy[n-1] - energy[n])/T[n]): # rejecting change
	    lattice[:, :, n] = lattice[:, :, n-1]
	    energy[n] = energy[n-1]



print("simulation done")
print(energy_f(lattice[:, :, -1]))


fig, ax1 = plt.subplots()

color = 'red'
ax1.set_xlabel('time')
ax1.set_ylabel('H', color=color)
ax1.plot(np.arange(timesteps), energy, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'blue'
ax2.set_ylabel('temperature', color=color)  # we already handled the x-label with ax1
ax2.plot(np.arange(timesteps), T, color=color)
ax2.tick_params(axis='y', labelcolor=color)

if not os.path.exists('testttt'):
    os.makedirs('testttt')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('testttt/energyplot.png')

#
# Calculate object size distribution
#


if save_plots:
    scale = timesteps/1000
    for n in range(1000):
        if n%5==0:
            print n
        plt.figure(n + 2, figsize=(6,6))
        lns = plt.plot()
        plt.axis([0, N+1, 0, N+1])
        plt.title("t=%d" % (n*scale))
        for i in range(N):
            for j in range(N):
                if lattice[i, j, n*scale] != -1.:
                    x, y = np.cos(lattice[i, j, n*scale]), np.sin(lattice[i, j, n*scale])
                    plt.arrow(i+0.5 - x*0.5, j+0.5 - y*0.5, x, y, length_includes_head=True, head_width=0.3)
        plt.savefig('testttt/lat_400lip_linear%03d.png'%n)
        plt.close(n)


np.save("testttt/final_lattice",lattice[:,:,-1])