#
#
#

import numpy as np
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation

N = 200 # number of x and y steps

dx = 1 # x-step of simulation
dy = dx # y-step of simulation
t_max = 10000 # max time of simulation

dt = 1 # time step of simulation, set to be stable (D*dt/dx^2 <= 0.25)
N_t = int(t_max/dt) # number of time steps 1 / (0.25 * 0.004 /1) = 1000 time steps

f = 0.035
k = 0.060
r = 10
c_v_init = 0.25
Dv = 0.08
Du = 0.16


def update_diffusion_term(matrix):
    ''' 
    Function that takes a concentration matrix and 
    returns the diffusion term for updating it for 1 time step,
    according to Gray-Scott model and finite-difference scheme.
    '''
    diffusion_term = (
        matrix[2:, 1:-1] +    # x+1; 2 because we need neigbours to the right 
        matrix[:-2, 1:-1] +   # x-1; 2 because interior point is Nx-2
        matrix[1:-1, 2:] +    # y+1
        matrix[1:-1, :-2] -   # y-1
        4 * matrix[1:-1, 1:-1]
    )
    return diffusion_term


def update_v_one_step(umatrix, vmatrix, f, D, dt, dx, k):
    ''' 
    Function that updates concentration matrix for substance V for 1 time step.
    Contains diffusion, reaction and decay terms according to Gray-Scott model.
    Includes reflecting (Von Neumann) boundary conditions.

    Params:
    - umatrix [np.ndarray]: the concentration matrix for substance U
    - vmatrix [np.ndarray]: the concentration matrix for substance V
    - f [float]: constant involved in decay of substance V
    - D [float]: diffusion constant for substance V
    - dt [float]: time step
    - dx [float]: spatial step in both x and y direction
    - k [float]: constant involved in decay of substance V

    Returns:
    - next_matrix [np.ndarray]: updated concentration matrix for substance V
    '''

    # set alpha constant
    alpha = D * dt / dx**2
    if alpha > 0.25:
        raise ValueError("Unstable constant: alpha value must be ≤ 0.25. Current value: ", alpha)
    
    next_matrix = vmatrix.copy()
    
    diffusion_term = update_diffusion_term(vmatrix)
    reaction_term = umatrix[1:-1, 1:-1] * vmatrix[1:-1,1:-1]**2
    decay_term = (f + k) * vmatrix[1:-1,1:-1]

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

    ''' 
    Function that updates concentration matrix for substance U for 1 time step.
    Contains diffusion, reaction and replenish terms according to Gray-Scott model.
    Includes reflecting (Von Neumann) boundary conditions.

    Params:
    - umatrix [np.ndarray]: the concentration matrix for substance U
    - vmatrix [np.ndarray]: the concentration matrix for substance V
    - f [float]: constant involved in replenishing of substance U
    - D [float]: diffusion constant for substance U
    - dt [float]: time step
    - dx [float]: spatial step in both x and y direction

    Returns:
    - next_matrix [np.ndarray]: updated concentration matrix for substance U
    '''

    # set alpha constant
    alpha = D * dt / dx**2
    if alpha > 0.25:
        raise ValueError("Unstable constant: alpha value must be ≤ 0.25. Current value: ", alpha)
    
    # determine max x and y shapes of matrix
    Ny, Nx = umatrix.shape
    next_matrix = umatrix.copy()
    
    diffusion_term = update_diffusion_term(umatrix)
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

def init_vmatrix(N, r, c):
    '''
    Function that initialises concentration matrix for substance V.
    Matrix will be all-zero values except for a square of rxr in the center,
    which will have cells with value c.
    
    Params:
    - N: size of concentration matrix for substance V (NxN)
    - r: length of square in centre
    - c: concentration for cells inside square
    
    Returns:
    - vmatrix: initialise concentration matrix for substance V.
    '''
    vmatrix = np.zeros((N,N))

    # create square
    start_y = N//2 - r//2
    end_y   = start_y + r
    start_x = N//2 - r//2
    end_x   = start_x + r

    vmatrix[start_y:end_y, start_x:end_x] = c
    return vmatrix


def run_gray_scott(N, r, c_v_init, f,Dv,Du,dt,dx,k):
    umatrix = np.full((N,N), 0.5)
    vmatrix = init_vmatrix(N, r, c_v_init)

    u_matrices = [umatrix]
    v_matrices = [vmatrix]

    # TODO: we might want to apply Strang splitting (operator splitting)
    for _ in range(N_t):
        umatrix = update_u_one_step(umatrix, vmatrix, f,Du,dt,dx)
        u_matrices.append(umatrix)
        vmatrix = update_v_one_step(umatrix, vmatrix, f,Dv,dt,dx,k)
        v_matrices.append(vmatrix)

    return u_matrices, v_matrices


def create_animation(matrices_over_time,  title):
    '''
    Creates animation for concentration over time.
    
    Params:
    - matrices_over_time: list of matrices, each matrix representing 1 timestep.
    - title: title string, for top of plot
    '''
    fig, ax = plt.subplots()

    ax.set_title(title)

    img = ax.imshow(
        matrices_over_time[0],
        vmin=0,
        vmax=1,
        origin='lower',
        extent=[0, 1, 0, 1]
    )

    fig.colorbar(img, ax=ax)

    def update(frame):
        img.set_data(matrices_over_time[frame])
        return img,

    anim = FuncAnimation(
        fig,
        update,
        frames=len(matrices_over_time),
        interval=10,
        blit=True
    )

    # anim.save("Figures/2.3/grayscott1.gif", fps=20)

    plt.show()
    plt.close(fig)


Du = 0.16
Dv = 0.08
f  = 0.026
k  = 0.051

u_matrices, v_matrices = run_gray_scott(N,r,c_v_init,f,Dv,Du,dt,dx,k)
create_animation(v_matrices, "V over time")
# create_animation(u_matrices, "U over time")