# 
# Part E (simulation part)
# 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

x_max = 1 # max x
y_max = 1 # max y
N = 100 # number of x and y steps

dx = x_max / N # x-step of simulation
dy = dx # y-step of simulation
t_max = 100 # max time of simulation
D = 0.0001 # diffusion coeff.

N_t = 1000 # number of time steps
dt = t_max / N_t # time step of simulation


# two dimensional
def one_step_2d(matrix):
    '''One diffusion step in the 2D diffusion simulation.
    Periodic boundaries on x.
    Set boundaries on y (1 at top, 0 on bottom).
    
    Params:
    - matrix: the matrix from which the next time step for diffusion is determined

    Returns:
    - next_matrix: the next time step diffusion matrix
    ''' 

    # set alpha constant
    alpha = D * dt / dx**2
    if alpha > 0.25:
        raise ValueError("Unstable constant: alpha value must be ≤ 0.25. Current value: ", alpha)
    
    # determine max x and y shapes of matrix
    Ny, Nx = matrix.shape
    next_matrix = matrix.copy()
    
    # loop over all matrix elements
    for j in range(1, Ny-1): 
        for i in range(0, Nx):
            next_value = (matrix[j, i] + alpha * (matrix[(j+1), i] + matrix[(j-1), i] +matrix[j, (i+1) % Nx] +matrix[j, (i-1) % Nx] -4 * matrix[j, i])) #i.e with periodic bounds in x
            next_matrix[j, i] = next_value

    # implement boundary conditions
    next_matrix[0, :] = 0 # put 0 at y=0
    next_matrix[-1, :] = 1 # put 1 at max y

    return next_matrix

def run_diffusion(current_matrix, N_t):
    ''' 
    Runs several timesteps for full diffusion simulation pipeline
    
    Params:
    - current_matrix: the initial conditions (initial matrix)
    - N_t: the number of time steps

    Returns:
    - matrices: a list of all matrices in the diffusion. Each matrix represents 1 timestep in the simulation.
    '''

    # create list to append to
    matrices = [current_matrix]
    
    # loop over time
    for _ in range(N_t-1):
        next_matrix = one_step_2d(current_matrix)
        matrices.append(next_matrix)
        current_matrix = next_matrix
    
    return matrices


def create_animation(matrices_over_time):
    '''
    Creates animation for diffusion over time.
    
    Params:
    - matrices_over_time: list of matrices, each matrix representing 1 timestep.
    '''

    fig = plt.figure()

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

    anim = FuncAnimation(fig, step, frames=len(matrices_over_time), interval=50, blit=True)
    plt.show()


def animate_diffusion(matrix_simulation, filename = "diffusion.mp4", fps=20):
    ''' 
    Runs diffusion simulation and creates animation for it.
    
    Params:
    - current_matrix: the initial conditions (initial matrix)
    - N_t: the number of time steps
    '''

    y = np.linspace(0, y_max, N)
    fig, ax = plt.subplots()

    selected = [0, 10, 50, 200, 500, len(matrix_simulation) - 1]
    for n in selected:
        ax.plot(y, matrix_simulation[n], label=f"t = {n*dt:.1f}", linewidth=1)

    line, = ax.plot(y, matrix_simulation[0][:,0])
   
    ax.set_ylim(0, 1)
    ax.set_xlabel('y')
    ax.set_title('Diffusion over time c(x,y,t)')

    def update(frame):
        line.set_ydata(matrix_simulation[frame][:,0])
        ax.set_title(f'Diffusion over time c(x,y,t) at time t={frame*dt:.2f} seconds')
        return line,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(matrix_simulation),
        interval=50,
        blit=True
    )
    ani.save(
        filename,
        writer="pillow",
        fps = fps
    )

    plt.close(fig)
    plt.show()

# set up initial matrix
first_matrix = np.zeros((N,N))
first_matrix[0, :] = 0
first_matrix[-1, :] = 1

# run simulation
diffusion_over_time = run_diffusion(first_matrix, N_t)
print(' number of time steps: ', len(diffusion_over_time))
animate_diffusion(diffusion_over_time, filename="diffusion.gif", fps=20)
create_animation(diffusion_over_time)

