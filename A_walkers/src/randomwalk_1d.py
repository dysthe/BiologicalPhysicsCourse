import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Parameters
ntime = 100
npart = 5000

# Generate random steps
steps = 3.24*(2 * np.random.randint(2, size=(ntime, npart)) - 1)
P = np.cumsum(steps, axis=0)

# Plot the random walk
plt.figure(1)
plt.plot(P)
plt.ylabel('Position')
plt.xlabel('Step Count')
plt.title('1-D Random Walk')

# Plot steps and cumulative positions for the first particle
plt.figure(6)
plt.subplot(2, 1, 1)
plt.plot(steps[:, 0], 'o-')
plt.subplot(2, 1, 2)
plt.plot(P[:, 0], 'o-')

# Histogram at ntime
plt.figure(2)
h = plt.hist(P[ntime - 1, :], bins=10)
counts, bins = h[0], h[1]
xlabel = (bins[:-1] + bins[1:]) / 2
xdist = xlabel
ydist = counts / npart / (bins[1] - bins[0])
plt.xlabel('Distance, steps')
plt.ylabel('Counts')

# Plot histogram and probability distribution
plt.figure(3)
plt.plot(xdist, ydist, '.-k')
plt.xlabel('Distance, steps')
plt.ylabel('Probability')
plt.title(f'Probability distribution after {ntime} steps')

# Gaussian distribution of width
x = np.arange(-ntime, ntime + 1, 1)
mu = 0
sigma = np.std(P[ntime - 1, :])
eta = (x - mu) / (sigma * np.sqrt(2))
G = np.exp(-eta ** 2) / (sigma * np.sqrt(2 * np.pi))
plt.plot(x, G, '-r')
plt.axis([-ntime / 2, ntime / 2, 0, max(G)])

# Cumulative probability distribution
plt.figure(4)
Ps = np.sort(P[ntime - 1, :])
plt.plot(Ps, (np.arange(npart) + 1) / npart, '-k')
plt.xlabel('Distance, X')
plt.ylabel('Cumulative probability P(x>X)')
plt.title('Cumulative probability distribution after 100 steps')

# Integral of Gaussian distribution of width
E = (1 + erf(eta)) / 2
plt.plot(x, E, '-r')
plt.axis([-ntime / 2, ntime / 2, 0, 1])

# Mean square displacement
SD = P ** 2
MSD = np.mean(SD, axis=1)
plt.figure(5)
t = np.arange(1, ntime + 1)

# Least-squares fit
b, stdb = np.polyfit(t, MSD, 1, full=False, cov=True)
slope = b[0]
std_slope = np.sqrt(stdb[0][0])
y = t * slope
plt.plot(t, MSD, 'ok', t, y, '-r')
plt.text(ntime / 2, max(MSD) / 2, f'slope = {slope:.2f} +/- {std_slope:.2f}')

# Show all plots
plt.show()
