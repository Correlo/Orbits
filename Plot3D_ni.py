import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

y10 = np.load('Results/NPY/disk10_ni.npy')
y100 = np.load('Results/NPY/disk100_ni.npy')
y1000 = np.load('Results/NPY/disk1000_ni.npy')


plt.close()
fig = plt.figure(figsize = (6,6))
ax = fig.add_subplot(111, projection = '3d')
for i in range(y10.shape[0]):

    ax.plot(y10[i,1,0].T, y10[i,1,1].T, y10[i,1,2].T, '--')

ax.set_xlim(-40,40)
ax.set_ylim(-40,40)
ax.set_zlim(-15,15)

ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
ax.set_zlabel('z [kpc]')

plt.savefig('Results/Figures/disk10_ni.png')

plt.close()
fig = plt.figure(figsize = (6,6))
ax = fig.add_subplot(111, projection = '3d')
for i in range(y100.shape[0]):

    ax.plot(y100[i,1,0].T, y100[i,1,1].T, y100[i,1,2].T, '--')

ax.set_xlim(-40,40)
ax.set_ylim(-40,40)
ax.set_zlim(-15,15)

ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
ax.set_zlabel('z [kpc]')

plt.savefig('Results/Figures/disk100_ni.png')

plt.close()
fig = plt.figure(figsize = (6,6))
ax = fig.add_subplot(111, projection = '3d')
for i in range(y1000.shape[0]):

    ax.plot(y1000[i,1,0].T, y1000[i,1,1].T, y1000[i,1,2].T, '--')

ax.set_xlim(-40,40)
ax.set_ylim(-40,40)
ax.set_zlim(-15,15)

ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
ax.set_zlabel('z [kpc]')

plt.savefig('Results/Figures/disk1000_ni.png')