import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize
# Set the size of the box
xsize = 1
ysize = 1
# Choose the number of time steps, number of particles, and how far the particles should move for each time step
timesteps = 100  # number of time steps
numpart = 200  # number of particles
stepsize = xsize / 100  # 1/500 of the box size makes it resemble the microscope images

# Initialize the vectors
xlist_all = np.zeros((timesteps, numpart))
ylist_all = np.zeros((timesteps, numpart))
poslist = np.zeros((timesteps * numpart, 3))
tvec = np.ones((numpart, 1))

# Set random start positions that are distributed over an area 4x the box area.
# This is done to ensure that approximately the same number of particles are always shown in the box.
xy = 2 * xsize * (np.random.rand(numpart, 2) - 0.5) + np.ones((numpart, 2)) * xsize / 2
xlist_all[0, :] = xy[:, 0]
ylist_all[0, :] = xy[:, 1]

# Find the particles that are inside the box
pnum = np.where((xy[:, 0] < xsize) & (xy[:, 0] > 0) & (xy[:, 1] < ysize) & (xy[:, 1] > 0))[0]
xyin = xy[pnum, :]
plistlen = len(pnum)

# Save positions of those inside the box
poslist[:plistlen, :] = np.hstack((xyin, np.zeros((plistlen, 1))))

# Create the figure and axis
fig, ax = plt.subplots()
scat = ax.scatter(xy[:, 0], xy[:, 1], c='k', edgecolor='k')
ax.set_xlim(0, xsize)
ax.set_ylim(0, ysize)
plt.title('Random walk with normally distributed random displacements')

def update(frame):
    global xy, plistlen
    # Generate normally distributed random displacements with mean 0 and standard deviation 1
    xymove = stepsize * np.random.randn(numpart, 2)
    xy = xy + xymove
    xlist_all[frame, :] = xy[:, 0]
    ylist_all[frame, :] = xy[:, 1]
    
    # Save positions of those inside the box
    pnum = np.where((xy[:, 0] < xsize) & (xy[:, 0] > 0) & (xy[:, 1] < ysize) & (xy[:, 1] > 0))[0]
    xyin = xy[pnum, :]
    padd = len(pnum)
    poslist[plistlen:plistlen + padd, :] = np.hstack((xyin, frame * np.ones((padd, 1))))
    plistlen += padd
    
    # Update plot
    scat.set_offsets(xy)
    return scat,

ani = animation.FuncAnimation(fig, update, frames=range(1, timesteps), blit=True, repeat=False)
plt.show()

# Analyze particle trajectories of all particles (not just those inside the box)
# Subtract the starting position of each particle and calculate the squared displacement of all particles
x0 = xlist_all - np.ones((timesteps, 1)) @ xlist_all[0, :][np.newaxis, :]
xsq = x0 ** 2
y0 = ylist_all - np.ones((timesteps, 1)) @ ylist_all[0, :][np.newaxis, :]
ysq = y0 ** 2

plt.figure(2)
msd_x = np.mean(xsq, axis=1) / stepsize ** 2
msd_y = np.mean(ysq, axis=1) / stepsize ** 2
msd_xy = np.mean(xsq + ysq, axis=1) / stepsize ** 2
t = np.arange(1, timesteps + 1)
plt.plot(t, msd_x, 'b', label=f'<x^2 d<x^2>/dt={np.polyfit(t, msd_x, 1)[0]}')
plt.plot(t, msd_y, 'g', label=f'<y^2>, d<y^2>/dt={np.polyfit(t, msd_y, 1)[0]}')
plt.plot(t, msd_xy, 'r', label=f'<x^2+y^2>, d<x^2+y^2>/dt={np.polyfit(t, msd_xy, 1)[0]}')

plt.xlabel('timesteps')
plt.ylabel('Mean square displacement, stepsize^2')
plt.legend(loc='upper left')
plt.title('Random walk with normally distributed random displacements')
plt.tight_layout()
plt.show()
