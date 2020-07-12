import numpy as np
from configparser import ConfigParser
from time import time
from sklearn.neighbors import KDTree

''' Fixed params '''

G = 4.302e-6               # Gravitational constant in Gadget units
Msun = 1.989e33            # [g]
kpc  = 3.085678e21         # kiloparsec [cm]
v    = 1e5                 # velocity units [cm/s]


def M_NFW(rho0, Rs, Rmax):

    C  = 4*np.pi*rho0*Rs**3     # Constant
    T1 = np.log((Rs + Rmax)/Rs) # First term
    T2 = Rmax/(Rs + Rmax)       # Second term

    return C*(T1 - T2)

def F(x, y):

    global M, m, rs, j, inter_p, KDt

    yf = np.zeros_like(y[j, :, :])

    if inter_p and KDt:

        global J, d, tree, k

        # Distance vector
        d_vec = y[j, 1, :] - y[J, 1, :]
        d_vecM = y[j, 1, :]  # Halo
        # Distace array
        d = np.sqrt(d**2 + ls**2) + 1e-16
        dM = np.sqrt(y[j,1,0]**2 + y[j,1,1]**2 + y[j,1,2]**2) + 1e-16  # Halo
        # Mass of the objects
        mk = m[J]

        # Eqdiff from higher to lower order
        yf[0,:] = - G*(np.einsum('i,ij', mk/d**3, d_vec) + M/(dM**3)*d_vecM)
        yf[1,:] = y[j,0,:]


    elif inter_p:

        # auxiliar index array
        index = np.arange(y.shape[0])

        # Distance vector
        d_vec = y[j,1,:] - y[index!=j,1,:]
        d_vecM = y[j,1,:] # Halo
        # Distace array
        d = np.sqrt(d_vec[:,0]**2 + d_vec[:,1]**2 + d_vec[:,2]**2 + ls**2) + 1e-16
        dM = np.sqrt(y[j,1,0]**2 + y[j,1,1]**2 + y[j,1,2]**2) + 1e-16 # Halo
        # Mass of the planets
        mk = m[index!=j]
        # Eqdiff from higher to lower order
        yf[0,:] = - G*(np.einsum('i,ij', mk/d**3, d_vec) + M/(dM**3)*d_vecM)
        yf[1,:] = y[j,0,:]

    else:

        d_vecM = y[j, 1, :]  # Halo
        dM = np.sqrt(y[j,1,0]**2 + y[j,1,1]**2 + y[j,1,2]**2) + 1e-16  # Halo
        yf[0,:] = - G*M/(dM**3)*d_vecM
        yf[1,:] = y[j,0,:]

    return yf


def Euler(x, y, dx, func):

    global j

    return y[j] + dx*func(x, y)


def RK4(x, y, dx, func):

    global j

    # Define k_i functions
    k_1 = func(x, y)
    k_2 = func(x, y + k_1*dx/2)
    k_3 = func(x, y + k_2*dx/2)
    k_4 = func(x, y + k_3*dx)

    # y_i+1
    yf = y[j] + dx/6*(k_1 + 2*k_2 + 2*k_3 + k_4)

    return yf



''' Parameters '''

#Read params.ini
params = ConfigParser()
params.sections()
params.read('params.ini')
Params = params['params']

datafile = Params['datafile']

# NFW mass parameters
rho0 = float(Params['rho0']) # characteristic density
Rs = float(Params['Rs'])     # scale length

t_i = float(Params['t_i']) # Initial time in years
t_f = float(Params['t_f']) # Last time in years
N = int(Params['N'])       # Number of time steps

ls = float(Params['ls']) # Softening lenght
inter_p = bool(int(Params['inter_p'])) # Body's interactions
dsch = Params['dsch'] # Derivative scheme

KDt = bool(int(Params['KDt'])) # Use KD-tree algoritm
k = int(Params['k']) + 1       # Number of neighbours
nl = int(Params['nl'])         # Number of leaves

Outnpy = Params['Outnpy'] # Output file .npy

''' Initial conditions '''
m, x, y, z, vx, vy, vz = np.loadtxt(datafile, unpack = True)
m = np.array(m, ndmin = 1)
m0 = m.copy() # copy of m

# Initial position
r0 = np.zeros((m.shape[0], 3))
r0[:,0] = x
r0[:,1] = y
r0[:,2] = z

# Initial velocity
v0      = np.zeros((m.shape[0], 3))
v0[:,0] = vx
v0[:,1] = vy
v0[:,2] = vz


''' Code '''

# time array
t, dt = np.linspace(t_i, t_f, N + 1, retstep = True)

# Def y array
y = np.zeros((len(m), 2, 3, N + 1))
y[:,0,:,0] = v0
y[:,1,:,0] = r0

print('The integration starts!')
t0 = time() # Initial time

for i in range(N):

        if KDt:

            # Tree scheme implementation to find closer neighbors
            tree = KDTree(y[:,1,:,i], leaf_size = nl)

        for j in range(m.shape[0]):

            if KDt:

                # Neighbours with force contribution
                d, J = tree.query([y[j, 1, :,i]], k=11)
                d = d[0, 1:]
                J = J[0, 1:]

            # Compute halo internal mass
            R = np.sqrt(y[j, 1, 0, i]**2 + y[j, 1, 1, i]**2)
            M = M_NFW(rho0, Rs, R)

            # Compute derivative
            if dsch == 'RK4':

                y[j,:,:,i+1] = RK4(t[i], y[:,:,:,i], dt, F)

            elif dsch == 'Euler':

                y[j,:,:,i+1] = Euler(t[i], y[:,:,:,i], dt, F)


t1 = time() # Final time

print('Saving data in %s' % Outnpy)
np.save(Outnpy, y)

print('Process complete with success')
print('Time of performance %.4f s' % (t1 - t0))
