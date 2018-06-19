import os

import numpy as np
import matplotlib.pyplot as plt

from utils import calculate_sizes

save_plots = True

timesteps = 100000
N = 100
N_lipids = 700


lattice = np.ones((N, N, timesteps)) * -1.

T = np.zeros(timesteps)
T_init = 1.
T[0] = T_init
T_decrease = 'linear'

density = float(N_lipids)/(N*N)
print("timesteps: ", timesteps, "N: ", N, "N_lipids: ",N_lipids)
print("density: ", density )

for i in range(N_lipids):
    lattice[np.random.randint(1, N-1), np.random.randint(1, N-1), 0] = np.random.random() * 2 * np.pi


def energy_f(lat):
    H_lat = (lat[1:N-1, 1:N-1] != -1.) * ((lat[0:N-2, 1:N-1] != -1.) * np.cos(lat[1:N-1, 1:N-1] - lat[0:N-2, 1:N-1]) * np.cos(2*lat[0:N-2, 1:N-1]) + 
                                          (lat[2:N, 1:N-1] != -1.) * np.cos(lat[1:N-1, 1:N-1] - lat[2:N, 1:N-1]) * np.cos(2*lat[2:N, 1:N-1]) + 
                                          (lat[1:N-1, 0:N-2] != -1.) * np.cos(lat[1:N-1, 1:N-1] - lat[1:N-1, 0:N-2]) * -np.cos(2*lat[1:N-1, 0:N-2]) + 
                                          (lat[1:N-1, 2:N] != -1.) * np.cos(lat[1:N-1, 1:N-1] - lat[1:N-1, 2:N]) * -np.cos(2*lat[1:N-1, 2:N]) +
                                          (lat[0:N-2, 0:N-2] != -1.)*np.cos(lat[1:N-1, 1:N-1] - lat[0:N-2, 0:N-2]) * np.cos(2*(lat[0:N-2, 0:N-2] - 0.25*np.pi)) + 
                                          (lat[2:N, 2:N] != -1.)*np.cos(lat[1:N-1, 1:N-1] - lat[2:N, 2:N]) * np.cos(2*(lat[2:N, 2:N] - 0.25*np.pi)) + 
                                          (lat[2:N, 0:N-2] != -1.)*np.cos(lat[1:N-1, 1:N-1] - lat[2:N, 0:N-2]) * np.cos(2*(lat[2:N, 0:N-2] - 0.75*np.pi)) + 
                                          (lat[0:N-2, 2:N] != -1.)*np.cos(lat[1:N-1, 1:N-1] - lat[0:N-2, 2:N]) * np.cos(2*(lat[2:N, 0:N-2] - 0.75*np.pi)) +
                                          (lat[0:N-2, 1:N-1] == -1.) * np.cos(lat[1:N-1, 1:N-1] - np.pi) + 
                                          (lat[2:N, 1:N-1] == -1.) * np.cos(lat[1:N-1, 1:N-1] - 2*np.pi) + 
                                          (lat[1:N-1, 0:N-2] == -1.) * np.cos(lat[1:N-1, 1:N-1] - 1.5*np.pi) + 
                                          (lat[1:N-1, 2:N] == -1.) * np.cos(lat[1:N-1, 1:N-1] - 0.5*np.pi) +
                                          (lat[0:N-2, 0:N-2] == -1.)*np.cos(lat[1:N-1, 1:N-1] - 1.25*np.pi) + 
                                          (lat[2:N, 2:N] == -1.)*np.cos(lat[1:N-1, 1:N-1] - 0.25*np.pi) + 
                                          (lat[2:N, 0:N-2] == -1.)*np.cos(lat[1:N-1, 1:N-1] - 1.75*np.pi) + 
                                          (lat[0:N-2, 2:N] == -1.)*np.cos(lat[1:N-1, 1:N-1] - 0.75*np.pi)
                                         )
    return np.sum(H_lat)


energy = np.zeros(timesteps)
energy[0] = energy_f(lattice[:, :, 0])

for n in range(1, timesteps):
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
    energy[n] = energy_f(lattice[:, :, n])
    if T_decrease == 'linear':
        T[n] = T[n-1] - T_init/(timesteps)
    if np.random.random() >= np.e**((energy[n-1] - energy[n])/T[n]): # rejecting change
	    lattice[:, :, n] = lattice[:, :, n-1]
	    energy[n] = energy[n-1]



print("simulation done")
print(energy_f(lattice[:, :, -1]))

# plt.figure(0, figsize=(6,6))
# lns = plt.plot(np.arange(timesteps), energy, "-")
# plt.xlabel("t")
# plt.ylabel("H")
# plt.savefig('energy_plot%s_%s_%s.png'%(N,N_lipids, timesteps))
# plt.close(0)

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

if not os.path.exists('lattice_plots_%s_%s_%s_%s_%s'%(N,N_lipids, timesteps, density, T_decrease)):
    os.makedirs('lattice_plots_%s_%s_%s_%s_%s'%(N,N_lipids, timesteps, density, T_decrease))

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('lattice_plots_%s_%s_%s_%s_%s/energy_plot.png'%(N,N_lipids, timesteps, density, T_decrease))

#
# Calculate object size distribution
#
lattice_snapshot = lattice[:,:,-1]
fname = "lattice_plots_%s_%s_%s_%s_%s/final_hist"%(N,N_lipids, timesteps,density,T_decrease)
calculate_sizes(lattice_snapshot, fname)

lattice_snapshot = lattice[:,:,1]
fname = "lattice_plots_%s_%s_%s_%s_%s/inital_hist"%(N,N_lipids, timesteps,density,T_decrease)
calculate_sizes(lattice_snapshot, fname)

if save_plots:
    scale = timesteps/1000
    for n in range(1000):
        plt.figure(n + 2, figsize=(6,6))
        lns = plt.plot()
        plt.axis([0, N+1, 0, N+1])
        plt.title("t=%d" % (n*scale))
        for i in range(N):
            for j in range(N):
                if lattice[i, j, n*scale] != -1.:
                    x, y = np.cos(lattice[i, j, n*scale]), np.sin(lattice[i, j, n*scale])
                    plt.arrow(i+0.5 - x*0.5, j+0.5 - y*0.5, x, y, length_includes_head=True, head_width=0.3)
        plt.savefig('lattice_plots_%s_%s_%s_%s_%s/lattice_%03d.png'%(N,N_lipids, timesteps,density,T_decrease,n))
        plt.close(n)


np.save("lattice_plots_%s_%s_%s_%s_%s/final_lattice"%(N,N_lipids, timesteps,density,T_decrease),lattice[:,:,-1])
