#
#
#

import numpy as np
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
import os

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

def l2_norm_u(diff_term, reaction_term, replenish_term, matrix, Du, dx):
    time_derivative = (Du/dx**2) * diff_term - reaction_term + replenish_term
    dy = dx
    return np.sum(matrix[1:-1,1:-1] * (time_derivative)) * dy * dx #Riemann summ

def l2_norm_v(diff_term, reaction_term, decay_term, matrix, Dv, dx):
    time_derivative = (Dv/dx**2) * diff_term + reaction_term - decay_term
    dy = dx
    return np.sum(matrix[1:-1,1:-1] * (time_derivative)) * dy * dx #Riemann summ

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
    reaction_term = (umatrix[1:-1, 1:-1] * vmatrix[1:-1,1:-1]**2)
    decay_term = ((f + k) * vmatrix[1:-1,1:-1])

    next_matrix[1:-1, 1:-1] = (
        vmatrix[1:-1, 1:-1]
        + alpha * diffusion_term
        + dt * reaction_term
        - dt * decay_term
    )

    #SOR version
    # step 1: update 'red' cells
    # step 2: update 'black' cells
    # step 3: combine into next_matrix

    # implement reflecting boundary conditions
    # here, vmatrix(-1,j) = vmatrix(1,j) etc.
    next_matrix[0, :]  = vmatrix[1, :]
    next_matrix[-1, :] = vmatrix[-2, :]
    next_matrix[:, 0]  = vmatrix[:, 1]
    next_matrix[:, -1] = vmatrix[:, -2]

    l2_norm = l2_norm_v(diffusion_term, reaction_term, decay_term, next_matrix, Dv, dx)
    return next_matrix, l2_norm

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
    reaction_term = (umatrix[1:-1, 1:-1] * vmatrix[1:-1, 1:-1]**2)
    replenish_term = ( f * (np.ones((Ny-2, Nx-2)) - umatrix[1:-1, 1:-1]))

    next_matrix[1:-1, 1:-1] = (
        umatrix[1:-1, 1:-1]
        + alpha * diffusion_term
        - dt * reaction_term
        + dt * replenish_term
    )

    # implement reflecting boundary conditions
    # here, umatrix(-1,j) = umatrix(1,j) etc.
    next_matrix[0, :]  = umatrix[1, :]
    next_matrix[-1, :] = umatrix[-2, :]
    next_matrix[:, 0]  = umatrix[:, 1]
    next_matrix[:, -1] = umatrix[:, -2]

    l2_norm = l2_norm_u(diffusion_term, reaction_term, replenish_term, next_matrix, Dv, dx)
    return next_matrix, l2_norm

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
    unorms = []
    vnorms = []

    # TODO: we might want to apply Strang splitting (operator splitting)
    for _ in range(N_t):
        umatrix, unorm = update_u_one_step(umatrix, vmatrix, f,Du,dt,dx)
        u_matrices.append(umatrix)
        unorms.append(unorm)

        vmatrix, vnorm = update_v_one_step(umatrix, vmatrix, f,Dv,dt,dx,k)
        v_matrices.append(vmatrix)
        vnorms.append(vnorm)

    return u_matrices, v_matrices, unorms, vnorms


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

def plot_last_frame(matrix, title):
    """
    Plots the last matrix as a heatmap
    and saves it to Figures/2.3 as a PNG.
    """

    save_dir = "Figures/2.3"
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    im = plt.imshow(matrix, vmin=0, vmax=1)
    
    cbar = plt.colorbar(im)
    cbar.set_label("Concentration")

    plt.title(title)

    filename = title.replace(" ", "_") + ".png"
    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved heatmap to {save_path}")


def plot_l2_norm_over_time(norms, dt, title="L² Norm Over Time"):
    """
    Plots the L² norm of a system over time.

    Params:
    - l2_norms [list or np.ndarray]: L² norm values at each timestep
    - dt [float]: time step size
    - title [str]: plot title
    """
    times = [i * dt for i in range(len(norms))]

    plt.figure(figsize=(6,4))
    plt.plot(times, norms, color='blue', lw=2)
    plt.xlabel("Time")
    plt.ylabel("L² Norm")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Mitosis
Du = 0.14
Dv = 0.06
f  = 0.035
k  = 0.065
u_matrices, v_matrices, unorms, vnorms = run_gray_scott(N,r,c_v_init,f,Dv,Du,dt,dx,k)
# create_animation(v_matrices, "V over time")
# create_animation(u_matrices, "U over time")
plot_last_frame(v_matrices[-1], title=f"Concentration of V at t=10000 for ICs: \n Du={Du}, Dv={Dv}, f={f}, k={k}")
plot_l2_norm_over_time((unorms), dt, title="L² Norm of U Over Time (Mitosis)")
plot_l2_norm_over_time(vnorms, dt, title="L² Norm of V Over Time (Mitosis)")

# Coral Pattern
Du = 0.16
Dv = 0.08
f  = 0.060
k  = 0.062
u_matrices, v_matrices, unorms, vnorms = run_gray_scott(N,r,c_v_init,f,Dv,Du,dt,dx,k)
# create_animation(v_matrices, "V over time")
plot_last_frame(v_matrices[-1], title=f"Concentration of V at t=10000 for ICs: \n Du={Du}, Dv={Dv}, f={f}, k={k}")
plot_l2_norm_over_time(unorms, dt, title="L² Norm of U Over Time (Coral)")
plot_l2_norm_over_time(vnorms, dt, title="L² Norm of V Over Time (Coral)")

# Spirals
Du, Dv, f, k = 0.12, 0.08, 0.020, 0.050
u_matrices, v_matrices, unorms, vnorms = run_gray_scott(N,r,c_v_init,f,Dv,Du,dt,dx,k)
# create_animation(v_matrices, "V over time")
plot_last_frame(v_matrices[-1], title=f"Concentration of V at t=10000 for ICs: \n Du={Du}, Dv={Dv}, f={f}, k={k}")
plot_l2_norm_over_time(unorms, dt, title="L² Norm of U Over Time (Spiral)")
plot_l2_norm_over_time(vnorms, dt, title="L² Norm of V Over Time (Spiral)")

# Zebra fish
Du, Dv, f, k = 0.16, 0.08, 0.035, 0.060
u_matrices, v_matrices, unorms, vnorms = run_gray_scott(N,r,c_v_init,f,Dv,Du,dt,dx,k)
# create_animation(v_matrices, "V over time")
plot_last_frame(v_matrices[-1], title=f"Concentration of V at t=10000 for ICs: \n Du={Du}, Dv={Dv}, f={f}, k={k}")
plot_l2_norm_over_time(unorms, dt, title="L² Norm of U Over Time (Zebra)")
plot_l2_norm_over_time(vnorms, dt, title="L² Norm of V Over Time (Zebra)")

