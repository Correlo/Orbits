To configure the code:

Modify params.ini fields to change the configuration of the simulation:

datafile: path of the input data. Format: m, x, y, z, vx, vy, vz 
rho0:    Characteristic density 
Rs:      Scale length
t_i:     Initial time
t_f:     Final time
N:       Number of time steps
ls:      Softening lenght 
inter_p: Activate Body's interactions
dsch:    Sélect dérivative scheme. Use Euler or RK4
KDt:     Activate KDtree
k:       Number of neighbors
nl:      Number of leaves
Outnpy:  Output file .npy

Params directory contains different configurations used in the project. All parameters must be in GADGET units.

To run the code: run Orbits_sys.py

The other codes allows to create the figures and perform some calculations.

Code developed by Martín Manuel Gómez Míguez