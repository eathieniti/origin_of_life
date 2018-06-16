
import numpy as np
import matplotlib.pyplot as plt

lattice = np.zeros((40, 40, 100000))
T = .1

for i in range(300):
    lattice[np.random.randint(1, 39), np.random.randint(1, 39), 0] = np.random.randint(1, 5)

def energy_f(lat, x, y):
    if lat[x, y] == 0:
        a, b = 0, 0
    elif lat[x, y] == 1:
        if lat[x+1, y] == 2 or lat[x+1, y] == 4:
            a = -1.
        elif lat[x+1, y] == 0:
            a = 0.
        else:
            a = 1.
        if lat[x, y-1] == 3 or lat[x, y-1] == 4:
            b = -1.
        elif lat[x, y-1] == 0:
            b = 0.
        else:
            b = 1.
    elif lat[x, y] == 2:
        if lat[x-1, y] == 1 or lat[x-1, y] == 3:
            a = -1.
        elif lat[x-1, y] == 0:
            a = 0.
        else:
            a = 1.
        if lat[x, y-1] == 3 or lat[x, y-1] == 4:
            b = -1.
        elif lat[x, y-1] == 0:
            b = 0.
        else:
            b = 1.
    elif lat[x, y] == 3:
        if lat[x+1, y] == 2 or lat[x+1, y] == 4:
            a = -1.
        elif lat[x+1, y] == 0:
            a = 0.
        else:
            a = 1.
        if lat[x, y+1] == 1 or lat[x, y+1] == 2:
            b = -1.
        elif lat[x, y+1] == 0:
            b = 0.
        else:
            b = 1.
    elif lat[x, y] == 4:
        if lat[x-1, y] == 1 or lat[x-1, y] == 3:
            a = -1.
        elif lat[x-1, y] == 0:
            a = 0.
        else:
            a = 1.
        if lat[x, y+1] == 1 or lat[x, y+1] == 2:
            b = -1.
        elif lat[x, y+1] == 0:
            b = 0.
        else:
            b = 1.
    return a + b

def energy_f2(lat):
    H_lat = (-1. * np.logical_or(
                     np.logical_and(np.logical_or(lat[1:39, 1:39] == 1, lat[1:39, 1:39] == 3), np.logical_or(lat[2:40, 1:39] == 2, lat[2:40, 1:39] == 4)), 
                     np.logical_and(np.logical_or(lat[1:39, 1:39] == 2, lat[1:39, 1:39] == 4), np.logical_or(lat[0:38, 1:39] == 1, lat[0:38, 1:39] == 3))
                   ) +
             1. * np.logical_or(
                     np.logical_and(np.logical_or(lat[1:39, 1:39] == 1, lat[1:39, 1:39] == 3), np.logical_or(lat[2:40, 1:39] == 1, lat[2:40, 1:39] == 3)), 
                     np.logical_and(np.logical_or(lat[1:39, 1:39] == 2, lat[1:39, 1:39] == 4), np.logical_or(lat[0:38, 1:39] == 2, lat[0:38, 1:39] == 4))
                  ) + 
             -1. * np.logical_or(
                     np.logical_and(np.logical_or(lat[1:39, 1:39] == 1, lat[1:39, 1:39] == 2), np.logical_or(lat[1:39, 0:38] == 3, lat[1:39, 0:38] == 4)),
                     np.logical_and(np.logical_or(lat[1:39, 1:39] == 3, lat[1:39, 1:39] == 4), np.logical_or(lat[1:39, 2:40] == 1, lat[1:39, 2:40] == 2))
                   ) + 
             1. * np.logical_or(
                     np.logical_and(np.logical_or(lat[1:39, 1:39] == 1, lat[1:39, 1:39] == 2), np.logical_or(lat[1:39, 0:38] == 1, lat[1:39, 0:38] == 2)),
                     np.logical_and(np.logical_or(lat[1:39, 1:39] == 3, lat[1:39, 1:39] == 4), np.logical_or(lat[1:39, 2:40] == 3, lat[1:39, 2:40] == 4))
                  ))
    return np.sum(H_lat)

def energy_f3(lat):
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

energy = np.zeros(100000)
energy[0] = energy_f3(lattice[:, :, 0])

for n in range(1, 100000):
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
    energy[n] = energy_f3(lattice[:, :, n])
    if np.random.random() >= np.e**((energy[n-1] - energy[n])/T): # rejecting change
	    lattice[:, :, n] = lattice[:, :, n-1]
	    energy[n] = energy[n-1]
	    

print "simulation done"
print sum(energy_f(lattice[:, :, -1], x, y) for x in range(1, 39) for y in range(1, 39))
print energy_f2(lattice[:, :, -1])
print energy_f3(lattice[:, :, -1])

plt.figure(0, figsize=(6,6))
lns = plt.plot(np.arange(100000), energy, "-")
plt.xlabel("t")
plt.ylabel("H")
plt.savefig('energy_plot.png')
plt.close(0)


for n in range(1000):  
    plt.figure(n + 2, figsize=(6,6))
    lns = plt.plot()
    plt.axis([0, 41, 0, 41])
    plt.title("t=%05d" % (n*100))
    for i in range(40):
        for j in range(40):
            if lattice[i, j, n*100] == 1:
                plt.arrow(i, j+1, 0.8, -0.8, length_includes_head=True, head_width=0.3)
            elif lattice[i, j, n*100] == 2:
                plt.arrow(i+1, j+1, -0.8, -0.8, length_includes_head=True, head_width=0.3)
            elif lattice[i, j, n*100] == 3:
                plt.arrow(i, j, 0.8, 0.8, length_includes_head=True, head_width=0.3)
            elif lattice[i, j, n*100] == 4:
                plt.arrow(i+1, j, -0.8, 0.8, length_includes_head=True, head_width=0.3)
    plt.savefig('lattice_plots/lattice_%03d.png' % n)
    plt.close(n)

