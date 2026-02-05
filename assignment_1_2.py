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
D = 0.0005 # diffusion coeff.

N_t = 100 # number of time steps
dt = t_max / N_t # time step of simulation


# set up (y,t) matrix
A = np.zeros((N, N_t)) # create matrix with all zeros
A[:, -1] =  1 # at max y value, put all 1's

# create empty matrix to append to
B = np.empty((N,N_t))

# two dimensional, but only depends on y because of symmetry

def one_step(row):
    # We calculate the y-values for the next time step.
    # We don't calculate x-values because across the whole x-axis, these will be the same
    # for the same value of y.

    alpha = D * dt / dx**2
    if alpha > 0.5:
        raise ValueError("Unstable constant: alpha value must be ≤ 0.5. Current value: ", alpha)
    
    next_row = [row[i] + alpha * (-2*row[i] + row[i+1] + row[i-1]) for i in range(1,len(row)-1)] #i.e without bounds

    # calculate for periodic bounds (wrong)
    # next_row_left = row[0] + (dt * D)/(dt**2) * (-2*row[0] + row[1] + row[-1])
    # next_row_right = row[-1] + (dt * D)/(dt**2) * (-2*row[-1] + row[0] + row[-2])

    # implement boundary conditions
    next_row.insert(0,0) # put zero at y=0
    next_row.append(1) # put 1 at max y

    return np.array(next_row)

def run_diffusion(current_row, N_t):

    # create empty matrix to append to
    B = [current_row]
    
    # loop over time (rows)
    for _ in range(N_t-1):
        next_row = one_step(current_row)
        B.append(next_row)
        current_row = next_row
    
    # print(B)
    return np.vstack(B)


def include_x(B, N, N_t):
    ''' 
    Function where the y,t matrix is turned into several x,y matrices.
    Each row becomes a new matrix.
    Useful for plotting.
    '''

    B_over_time = []

    for i in range(N_t):
        row = B[i,:] # extract row i
        # repeat row N times into matrix
        matrix = np.tile(row, (N, 1))
        # append to list
        B_over_time.append(matrix)

    return B_over_time


def create_animation(matrices_over_time):
    animation = plt.figure()

    color_scale = plt.pcolormesh(
        matrices_over_time[0],
        vmin=0, # min diffusion
        vmax=1, # max diffusion
        shading="auto"
    )
    plt.colorbar()

    def step(i):
        color_scale.set_array(matrices_over_time[i].ravel())
        return color_scale,

    anim = FuncAnimation(animation, step, interval=50, blit=True)
    plt.show()



first_row = np.zeros(N)
first_row[-1] = 1
diffusion_matrix = run_diffusion(first_row, N_t)
diffusion_over_time = include_x(diffusion_matrix, N, N_t)

print(' number of time steps: ', len(diffusion_over_time))
print(' shape of data of each time step: ', diffusion_over_time[0].shape)
print(' shape of data of each time step: ', diffusion_over_time[-1])

create_animation(diffusion_over_time)



