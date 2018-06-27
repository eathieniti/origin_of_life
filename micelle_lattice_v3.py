
import numpy as np
import matplotlib.pyplot as plt

timesteps = 1000000

lattice = np.zeros((40, 40, timesteps))

T = np.zeros(timesteps)
T[0] = .1

for i in range(300):
    lattice[np.random.randint(1, 39), np.random.randint(1, 39), 0] = np.random.randint(1, 5)


def energy_f(lat):
    H_lat = (-1. * (
                     ((lat[1:39, 1:39]==1) + (lat[1:39, 1:39]==3))*((lat[2:40, 1:39]==2) + (lat[2:40, 1:39]==4)) + 
                     ((lat[1:39, 1:39]==2) + (lat[1:39, 1:39]==4))*((lat[0:38, 1:39]==1) + (lat[0:38, 1:39]==3))
                   ) +
             1. * (
                     ((lat[1:39, 1:39]==1) + (lat[1:39, 1:39]==3))*((lat[2:40, 1:39]==1) + (lat[2:40, 1:39]==3)) + 
                     ((lat[1:39, 1:39]==2) + (lat[1:39, 1:39]==4))*((lat[0:38, 1:39]==2) + (lat[0:38, 1:39]==4))
                  ) + 
             -1. * (
                     ((lat[1:39, 1:39]==1) + (lat[1:39, 1:39]==2))*((lat[1:39, 0:38]==3) + (lat[1:39, 0:38]==4)) +
                     ((lat[1:39, 1:39]==3) + (lat[1:39, 1:39]==4))*((lat[1:39, 2:40]==1) + (lat[1:39, 2:40]==2))
                   ) + 
             1. * (
                     ((lat[1:39, 1:39]==1) + (lat[1:39, 1:39]==2))*((lat[1:39, 0:38]==1) + (lat[1:39, 0:38]==2)) +
                     ((lat[1:39, 1:39]==3) + (lat[1:39, 1:39]==4))*((lat[1:39, 2:40]==3) + (lat[1:39, 2:40]==4))
                  ))
    return np.sum(H_lat)


energy = np.zeros(timesteps)
energy[0] = energy_f(lattice[:, :, 0])

for n in range(1, timesteps):
    lattice[:, :, n] = lattice[:, :, n-1]
    while True:
        i, j = np.random.randint(1, 39), np.random.randint(1, 39)
        if lattice[i, j, n] != 0:
            break
    if np.random.random() < 0.5:
        lattice[i, j, n] = np.random.randint(1, 5)
    else:
        while True:
            di, dj = np.random.randint(-1, 2), np.random.randint(-1, 2)
            if 0 < i+di < 39 and 0 < j+dj < 39:
                break
        lattice[i, j, n], lattice[i+di, j+dj, n] = lattice[i+di, j+dj, n-1], lattice[i, j, n-1]
    energy[n] = energy_f(lattice[:, :, n])
    T[n] = T[n-1] #- 1./timesteps
#    print (energy[n-1] - energy[n])/T[n]
    if np.random.random() >= np.e**((energy[n-1] - energy[n])/T[n]): # rejecting change
	    lattice[:, :, n] = lattice[:, :, n-1]
	    energy[n] = energy[n-1]
	    

print "simulation done"
print energy_f(lattice[:, :, -1])

# plt.figure(0, figsize=(6,6))
# lns = plt.plot(np.arange(timesteps), energy, "-")
# plt.xlabel("t")
# plt.ylabel("H")
# plt.savefig('energy_plot.png')
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

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('energy_plot.png')

scale = timesteps/1000
for n in range(1000):  
    plt.figure(n + 2, figsize=(6,6))
    lns = plt.plot()
    plt.axis([0, 41, 0, 41])
    plt.title("t=%d" % (n*scale))
    for i in range(40):
        for j in range(40):
            if lattice[i, j, n*scale] == 1:
                plt.arrow(i, j+1, 0.8, -0.8, length_includes_head=True, head_width=0.3)
            elif lattice[i, j, n*scale] == 2:
                plt.arrow(i+1, j+1, -0.8, -0.8, length_includes_head=True, head_width=0.3)
            elif lattice[i, j, n*scale] == 3:
                plt.arrow(i, j, 0.8, 0.8, length_includes_head=True, head_width=0.3)
            elif lattice[i, j, n*scale] == 4:
                plt.arrow(i+1, j, -0.8, 0.8, length_includes_head=True, head_width=0.3)
    plt.savefig('lattice_plots/lattice_%03d.png' % n)
    plt.close(n)

