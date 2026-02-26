#
#
#

import numpy as np

x_max = 1 # max x
y_max = 1 # max y
N = 50 # number of x and y steps

dx = x_max / N # x-step of simulation 1/50 -> 0.02
dy = dx # y-step of simulation
t_max = 1 # max time of simulation
D = 1 # diffusion coeff.

dt = 0.25 * dx**2 / D # time step of simulation, set to be stable (D*dt/dx^2 <= 0.25)
N_t = int(t_max/dt) # number of time steps 1 / (0.25 * 0.004 /1) = 1000 time steps

f = 0.6


def update_v_one_step(umatrix, vmatrix, f, D, dt, dx, k):

    # set alpha constant
    alpha = D * dt / dx**2
    if alpha > 0.25:
        raise ValueError("Unstable constant: alpha value must be ≤ 0.25. Current value: ", alpha)
    
    # determine max x and y shapes of matrix
    Ny, Nx = vmatrix.shape
    next_matrix = vmatrix.copy()
    
    diffusion_term = (
        vmatrix[2:, 1:-1] +    # x+1; 2 because we need neigbours to the right 
        vmatrix[:-2, 1:-1] +   # x-1; 2 because interior point is Nx-2
        vmatrix[1:-1, 2:] +    # y+1
        vmatrix[1:-1, :-2] -   # y-1
        4 * vmatrix[1:-1, 1:-1]
    )

    reaction_term = umatrix[1:-1, 1:-1] * vmatrix[1:-1,1:-1]
    decay_term = (f + k) * vmatrix

    next_matrix[1:-1, 1:-1] = (
        vmatrix[1:-1, 1:-1]
        + alpha * diffusion_term
        + reaction_term
        - decay_term
    )

    # implement reflecting boundary conditions
    # here, vmatrix(-1,j) = vmatrix(1,j) etc.
    next_matrix[0, :]  = vmatrix[1, :]
    next_matrix[-1, :] = vmatrix[-2, :]
    next_matrix[:, 0]  = vmatrix[:, 1]
    next_matrix[:, -1] = vmatrix[:, -2]

    return next_matrix

def update_u_one_step(umatrix, vmatrix, f, D, dt, dx):

    # set alpha constant
    alpha = D * dt / dx**2
    if alpha > 0.25:
        raise ValueError("Unstable constant: alpha value must be ≤ 0.25. Current value: ", alpha)
    
    # determine max x and y shapes of matrix
    Ny, Nx = umatrix.shape
    next_matrix = umatrix.copy()
    
    diffusion_term = (
        umatrix[2:, 1:-1] +    # x+1; 2 because we need neigbours to the right 
        umatrix[:-2, 1:-1] +   # x-1; 2 because interior point is Nx-2
        umatrix[1:-1, 2:] +    # y+1
        umatrix[1:-1, :-2] -   # y-1
        4 * umatrix[1:-1, 1:-1]
    )

    reaction_term = umatrix[1:-1, 1:-1] * vmatrix[1:-1, 1:-1]**2
    replenish_term = f * (np.ones((Ny-2, Nx-2)) - umatrix[1:-1, 1:-1])

    next_matrix[1:-1, 1:-1] = (
        umatrix[1:-1, 1:-1]
        + alpha * diffusion_term
        - reaction_term
        + replenish_term
    )

    # implement reflecting boundary conditions
    # here, umatrix(-1,j) = umatrix(1,j) etc.
    next_matrix[0, :]  = umatrix[1, :]
    next_matrix[-1, :] = umatrix[-2, :]
    next_matrix[:, 0]  = umatrix[:, 1]
    next_matrix[:, -1] = umatrix[:, -2]

    return next_matrix
