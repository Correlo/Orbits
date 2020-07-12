import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import data
y_Eu = np.load('Results/NPY/Sun_Eu.npy')
y_RK4 = np.load('Results/NPY/Sun_RK4.npy')

t_i = 0
t_f = 2
N = 1000

t, dt = np.linspace(t_i, t_f, N + 1, retstep = True)

if y_Eu.shape[0] != y_RK4.shape[0]:

    raise ValueError('Error in the dimensions of input arrays')

def animate(i):

    global y_Eu, y_RK4, line_Eu, line_RK4, dot_Eu, dot_RK4, t

    line_Eu.set_data(y_Eu[0,1,0,:i], y_Eu[0,1,1,:i])
    line_RK4.set_data(y_RK4[0,1,0,:i], y_RK4[0,1,1,:i])
    dot_Eu.set_data(y_Eu[0,1,0,i], y_Eu[0,1,1,i])
    dot_RK4.set_data(y_RK4[0,1,0,i], y_RK4[0,1,1,i])
    title.set_text(r't = %.2f $t_{GADGET}$' % t[i])

    return line_Eu, line_RK4, dot_Eu, dot_RK4

def init():

    line_Eu.set_data([], [])
    line_RK4.set_data([], [])
    dot_Eu.set_data([], [])
    dot_RK4.set_data([], [])

    return line_Eu, line_RK4, dot_Eu, dot_RK4


fig, ax = plt.subplots(figsize = (6,6))
title = ax.set_title(r't = %.2f $t_{GADGET}$' % t[0])
# Eulerian sol
line_Eu, = ax.plot([], [], 'b--', label = 'Eu')
dot_Eu, = ax.plot([], [], 'b.')
# Runge-Kutta sol
line_RK4, = ax.plot([], [], 'r--', label = 'RK4')
dot_RK4, = ax.plot([], [], 'r.')
ax.set_xlim(-22, 22)
ax.set_ylim(-23, 24)
ax.axis('equal')


ax.legend(loc = 'upper right', frameon = False)

# This function takes as arguments:
#   * fig : the figure we will use for plotting purposes
#   * animate : the function that will be called each frame
#   * interval : time in ms between frames
#   * blit : clear the figure each frame
#   * fargs : extra arguments to 'animate'
#
# Please read carefully the documentation
ani = animation.FuncAnimation(fig, animate, y_Eu.shape[-1], init_func = init,
                              interval = 25, blit = True, repeat = False)
#plt.show()
ani.save('Results/Animations/Sun_path.gif', writer = 'imagemagick', fps = 25)
