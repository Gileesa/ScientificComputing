# 
# Part E (simulation part)
# 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_max = 1 # max x
y_max = 1 # max y
N = 100 # number of x and y steps

dx = x_max / N # x-step of simulation
dy = dx # y-step of simulation
t_max = 10 # max time of simulation
D = 0.00005 # diffusion coeff.

N_t = 100 # number of time steps
dt = t_max / N_t # time step of simulation


# set up (y,t) matrix
A = np.zeros((N, N_t)) # create matrix with all zeros
A[:, -1] =  1 # at max y value, put all 1's

# create empty matrix to append to
B = np.empty((N,N_t))

# two dimensional, but only depends on y because of symmetry

def one_step_2d(matrix):
    # We calculate the y-values for the next time step.
    # We don't calculate x-values because across the whole x-axis, these will be the same
    # for the same value of y.

    alpha = D * dt / dx**2
    if alpha > 0.25:
        raise ValueError("Unstable constant: alpha value must be ≤ 0.25. Current value: ", alpha)
    
    Ny, Nx = matrix.shape
    next_matrix = matrix.copy()
    
    for j in range(1, Ny-1): 
        for i in range(0, Nx):
            next_value = (matrix[j, i] + alpha * (matrix[(j+1), i] + matrix[(j-1), i] +matrix[j, (i+1) % Nx] +matrix[j, (i-1) % Nx] -4 * matrix[j, i])) #i.e with periodic bounds in x
            next_matrix[j, i] = next_value

    # implement boundary conditions
    next_matrix[0, :] = 0 # put 0 at y=0
    next_matrix[-1, :] = 1 # put 1 at max y

    return next_matrix

def run_diffusion(current_matrix, N_t):

    # create empty matrix to append to
    matrices = [current_matrix]
    
    # loop over time (rows)
    for _ in range(N_t-1):
        next_matrix = one_step_2d(current_matrix)
        matrices.append(next_matrix)
        current_matrix = next_matrix
    
    return matrices


def create_animation(matrices_over_time):
    animation = plt.figure()

    color_scale = plt.pcolormesh(
        matrices_over_time[0],
        vmin=0, # min heat
        vmax=1, # max heat
        shading="auto"
    )
    plt.colorbar()

    def step(i):
        color_scale.set_array(matrices_over_time[i].ravel())
        return color_scale,

    anim = FuncAnimation(animation, step, frames=len(matrices_over_time), interval=50, blit=True)
    plt.show()



first_matrix = np.zeros((N,N))
first_matrix[-1, :] = np.ones(N)
diffusion_over_time = run_diffusion(first_matrix, N_t)

print(' number of time steps: ', len(diffusion_over_time))
print(' shape of data of each time step: ', diffusion_over_time[0].shape)
print(' data of each time step: ', diffusion_over_time[-1])

create_animation(diffusion_over_time)