import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

x_max = 1 # max x
y_max = 1 # max y
N = 50 # number of x and y steps 

dx = x_max / N # x-step of simulation 1/50 -> 0.02
dy = dx # y-step of simulation
t_max = 1 # max time of simulation
D = 1 # diffusion coeff.


#dt = t_max / N_t # time step of simulation
dt = 0.25 * dx**2 / D # time step of simulation, set to be stable (D*dt/dx^2 <= 0.25)
N_t = int(t_max/dt) # number of time steps 1 / (0.25 * 0.004 /1) = 1000 time steps

def jacobi_iteration(c, max_iteration, epsilon = 10**(-5)):
    """
    This function performs the Jacobi iteration for solving the 2D Laplace equation with given boundary conditions.
    params:
    c: the initial matrix with boundary conditions set
    N: the number of x and y steps
    epsioln: the convergence criterion for the iteration
    returns:
    c: the matrix after convergence    

    array[y, x] -> array[row, column] -> array[j, i]
    """
    c_old = c.copy()
    c_new = c.copy()
    Ny, Nx = c.shape

    c_old[0, :] = 0 # set boundary condition
    c_old[-1, :] = 1 # set boundary condition

    delta_list = []

    for k in range(max_iteration):

        for j in range(1, Ny-1):
            for i in range(Nx):
                right = (i + 1) % Nx # periodic boundary condition 49 + 1 = 50 % 50 = 0
                left = (i - 1) % Nx # periodic boundary condition 0 - 1 = -1 % 50 = 49
            
                c_new[j, i] = 0.25 * (c_old[j + 1, i] + c_old[j - 1, i] + c_old[j, right] + c_old[j, left])
        c_new[0, :] = 0 # set boundary condition
        c_new[-1, :] = 1 # set boundary condition

        delta = np.max(np.abs(c_new - c_old)) 
        delta_list.append(delta)

        if delta < epsilon:
            print(f"Converged after {k} iterations.")
            return c_new, np.array(delta_list)
        
        c_old[:, :] = c_new[:, :]

    return c_new, np.array(delta_list)



  
      
    