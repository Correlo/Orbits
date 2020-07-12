import numpy as np
import matplotlib.pyplot as plt

y = np.load('Results/NPY/Sun_RK4.npy')

plt.close()
fig, ax = plt.subplots(1, 2, figsize = (12,6))
ax[0].set_title('Space')
ax[0].plot(y[:,1,0].T, y[:,1,1].T, '--')
ax[0].set_xlabel('x [kpc]')
ax[0].set_ylabel('y [kpc]')
ax[0].axis('equal')

ax[1].set_title('Velocity')
ax[1].plot(y[:,0,0].T, y[:,0,1].T, '--')
ax[1].set_xlabel(r'$v_x$ [km/s]')
ax[1].set_ylabel(r'$v_y$ [km/s]')
ax[1].axis('equal')

plt.savefig('Results/Figures/Sun_RK4.png')