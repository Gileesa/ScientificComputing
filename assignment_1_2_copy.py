# 
# Part E (simulation part)
# 
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

# animation for diffusion over time
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
        ax.plot(y, matrix_simulation[n][:,0], label=f"t = {n*dt:.1f}", linewidth=1)

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

# Part E analytical solution
def analytical_solution(y_array, t, D):
    ''' 
    Analytical solution for diffusion in 1D with boundary conditions c(0,t) = 0 and c(L,t) = 1.
    
    Params:
    - y: the y values at which to evaluate the solution
    - t: the time at which to evaluate the solution
    - D: the diffusion coefficient

    Returns:
    - c: the concentration values at each y for time t
    '''
    y_array = np.array(y_array)

    if t <= 0:
        output = np.zeros_like(y_array)
        output[-1] = 1 # set boundary condition at t=0
        output[0] = 0 # set boundary condition at t=0
        return output
    
    denominator = 2 * np.sqrt(D * t)
    profile = np.zeros_like(y_array)

    for j, y in enumerate(y_array):
        # compute the series sum for the analytical solution
        series_sum = 0
    
        for i in range(0, 1000):
            term_left = math.erfc((1 - y + 2 * i) / denominator)
            term_right = math.erfc((1 + y + 2 * i) / denominator)
            equation = term_left - term_right
            series_sum += equation

            if abs(equation) < 1e-10: # break if the term is small enough to not contribute to the sum
                break
        profile[j] = series_sum
    return profile

# Plot Question E -

def plot_analytical_solution(concentration_over_time):
    ''' 
    Plots the analytical solution for diffusion in 1D with boundary conditions c(0,t) = 0 and c(L,t) = 1 for different times.
    
    Params:
    - concentration_over_time: list of concentration fields over time
    '''
    y = np.linspace(0, y_max, N)
    times = [0, 0.001, 0.01, 0.1, 1.0]
   
    for time in times:
        frame = int((round(time/dt)))

        if frame >= len(concentration_over_time):
            frame = len(concentration_over_time) - 1
        
        concentration_field = concentration_over_time[frame]
        numerical_profile = concentration_field[:,0]
        analytical_profile = analytical_solution(y, time, D)
        plt.figure()
        plt.plot(y, numerical_profile, label=f"Numerical t={time:.3f} s", linestyle='dashed')
        plt.plot(y, analytical_profile, label=f"Analytical t={time:.3f} s", linestyle='solid')
    
        plt.xlabel('y')
        plt.ylabel('c(x,y,t)')
        plt.title('Analytical Solution of Diffusion over time')
        plt.legend()
        plt.savefig(f"analytical_solution_t{time:.3f}.png")
        plt.show()


# Question F

def diffusion(concentration_over_time):
    ''' 
    Runs diffusion simulation and creates animation for it.
    
    Params:
    - concentration_over_time: list of concentration fields over time
    '''
    times = [0, 0.001, 0.01, 0.1, 1.0]
    frame_indices = []

    for t in times:
        frame = int(t/dt)

        if frame >= len(concentration_over_time):
            frame = len(concentration_over_time) - 1
        frame_indices.append(frame)

    x = np.linspace(0, x_max, N)
    y = np.linspace(0, y_max, N)
    fig, axes = plt.subplots(1, len(times), figsize=(15, 3))
    shared_mappable = None
    
    for idx, ax in enumerate(axes):
        concentration_fields = concentration_over_time[frame_indices[idx]]
        shared_mappable = ax.imshow(
            concentration_fields,
            extent=[0, 1, 0, 1], 
            vmin=0,
            vmax=1,
            origin='lower'
        )
    
        ax.set_title(f't = {times[idx]:.3f} seconds')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 1)       
        ax.set_ylim(0, 1)

    fig.colorbar(shared_mappable, ax=axes, label ="concentration c(x,y,t)")
    
    
    plt.savefig("diffusion_over_time.png")
    plt.show()

# set up initial matrix
first_matrix = np.zeros((N,N))
first_matrix[0, :] = 0
first_matrix[-1, :] = 1

# run simulation
diffusion_over_time = run_diffusion(first_matrix, N_t)
print(' number of time steps: ', len(diffusion_over_time))
#animate_diffusion(diffusion_over_time, filename="diffusion.gif", fps=20)
#create_animation(diffusion_over_time)
diffusion(diffusion_over_time)
plot_analytical_solution(diffusion_over_time)
