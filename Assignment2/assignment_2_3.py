#
#
#

import numpy as np
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import os
import re

N = 200 # number of x and y steps

dx = 1 # x-step of simulation
dy = dx # y-step of simulation
t_max = 10000 # max time of simulation

dt = 1 # time step of simulation, set to be stable (D*dt/dx^2 <= 0.25)
N_t = int(t_max/dt) # number of time steps

f = 0.035
k = 0.060
r = 10
c_v_init = 0.25
Dv = 0.08
Du = 0.16


def update_diffusion_term(matrix:np.ndarray):
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

def l2_norm_u(diff_term:float, reaction_term:float, replenish_term:float, matrix:np.ndarray, Du:float, dx:float):
    time_derivative = (Du/dx**2) * diff_term - reaction_term + replenish_term
    dy = dx
    return np.sum(matrix[1:-1,1:-1] * (time_derivative)) * dy * dx #Riemann summ

def l2_norm_v(diff_term:float, reaction_term:float, decay_term:float, matrix:np.ndarray, Dv:float, dx:float):
    time_derivative = (Dv/dx**2) * diff_term + reaction_term - decay_term
    dy = dx
    return np.sum(matrix[1:-1,1:-1] * (time_derivative)) * dy * dx #Riemann summ

def update_v_one_step(umatrix: np.ndarray, vmatrix: np.ndarray, f:float, D:float, dt:float, dx:float, k:float):
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

    # TODO: SOR version
    # step 1: update 'red' cells
    # step 2: update 'black' cells
    # step 3: combine into next_matrix

    # implement reflecting boundary conditions
    # here, vmatrix(-1,j) = vmatrix(1,j) etc.
    next_matrix[0, :]  = next_matrix[1, :]
    next_matrix[-1, :] = next_matrix[-2, :]
    next_matrix[:, 0]  = next_matrix[:, 1]
    next_matrix[:, -1] = next_matrix[:, -2]

    l2_norm = l2_norm_v(diffusion_term, reaction_term, decay_term, next_matrix, Dv, dx)
    return next_matrix, l2_norm

def update_u_one_step(umatrix: np.ndarray, vmatrix: np.ndarray, f:float, D:float, dt:float, dx:float):

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
    next_matrix[0, :]  = next_matrix[1, :]
    next_matrix[-1, :] = next_matrix[-2, :]
    next_matrix[:, 0]  = next_matrix[:, 1]
    next_matrix[:, -1] = next_matrix[:, -2]

    l2_norm = l2_norm_u(diffusion_term, reaction_term, replenish_term, next_matrix, Dv, dx)
    return next_matrix, l2_norm

def init_vmatrix(N: int, r: float, c: float, add_noise):
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

    if add_noise:
        epsilon = 0.2  # noise strength
        noise = epsilon * (2*np.random.rand(r, r) - 1)
        vmatrix[start_y:end_y, start_x:end_x] = c + noise
    else:
        vmatrix[start_y:end_y, start_x:end_x] = c

    vmatrix = np.clip(vmatrix, 0, None)
    return vmatrix


def run_gray_scott(N: int, r: int, c_v_init: float, f: float,Dv: float,Du: float,dt: float,dx: float, k:float, N_t:int, add_noise:bool=False):
    '''
    Function that runs full Gray-Scott reaction-diffusion pipeline.

    Params:
    - N: number of grid points along one axis
    - r: length of one side of the square in the centre of the initial V-matrix
    - c_v_init: concentration of V in square in initial V-matrix
    - f: constant that controls rate of replenishing of U
    - Dv: diffusion constant for V
    - Du: diffusion constant for u
    - dt: time step size
    - dx: length of 1 grid side; dy=dx
    - k: constant that controls decay of U together with constant f
    '''
    umatrix = np.full((N,N), 0.5)
    vmatrix = init_vmatrix(N, r, c_v_init, add_noise)

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


def create_animation(matrices_over_time: list, title: str, save: bool = True, step: int = 5, as_gif: bool = True):
    """
    Creates animation for concentration over time and optionally saves it as GIF or MP4.
    
    Only every `step` frames are used to speed up saving.

    Params:
    - matrices_over_time: list of 2D numpy arrays (frames)
    - title: title for the plot
    - save: whether to save the animation
    - step: save every `step` frames
    - as_gif: if True, saves as GIF, else saves as MP4
    """
    # Subsample frames
    frames_to_use = matrices_over_time[::step]
    
    fig, ax = plt.subplots()
    ax.set_title(title)

    img = ax.imshow(
        frames_to_use[0],
        vmin=0,
        vmax=1,
        origin='lower',
        extent=[0, 1, 0, 1],
        cmap='viridis'
    )
    fig.colorbar(img, ax=ax)

    def update(frame):
        img.set_data(frames_to_use[frame])
        return img,

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames_to_use),
        interval=50,
        blit=True
    )

    plt.show()
    plt.close(fig)

    if save:
        save_dir = "Figures/2.3"
        os.makedirs(save_dir, exist_ok=True)
        filename = title.replace(" ", "_")
        if as_gif:
            save_path = os.path.join(save_dir, filename + ".gif")
            writer = PillowWriter(fps=20)
        else:
            save_path = os.path.join(save_dir, filename + ".mp4")
            from matplotlib.animation import FFMpegWriter
            writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        
        anim.save(save_path, writer=writer)
        print(f"Animation saved as {save_path}")

def clean_filename(title: str, ext: str = ".png") -> str:
    """
    Turn a title into a safe filename.
    Removes spaces, newlines, and most special characters.
    """
    # Remove all non-alphanumeric characters (keep _ and -)
    safe = re.sub(r'[^A-Za-z0-9_\-]', '_', title)
    return safe + ext

def plot_last_frame(matrix:np.ndarray, title: str):
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

    filename = clean_filename(title, ext=".png")
    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved heatmap to {save_path}")


def plot_l2_norm_over_time(norms, dt: float, title: str="L² Norm Over Time"):
    """
    Plots the L² norm of a system over time.

    Params:
    - norms [list or np.ndarray]: L² norm values at each timestep
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

def run_full(Du, Dv, f, k, N_t, c_v_init, r, dt,dx, shape:str, add_noise:bool=False, animation:bool=False):
    u_matrices, v_matrices, unorms, vnorms = run_gray_scott(N,r,c_v_init,f,Dv,Du,dt,dx,k, N_t, add_noise)
    if animation:
        create_animation(v_matrices, f"V over time ({shape})")
        create_animation(u_matrices, f"U over time ({shape})")
    plot_last_frame(v_matrices[-1], title=f"Concentration of V at t={N_t} for ICs \n Du={Du}, Dv={Dv}, f={f}, k={k} ({shape})")
    # plot_l2_norm_over_time((unorms), dt, title=f"L² Norm of U Over Time {shape}")
    # plot_l2_norm_over_time(vnorms, dt, title=f"L² Norm of V Over Time {shape}")
    # total_norm = np.array(unorms) + np.array(vnorms)
    # plot_l2_norm_over_time(total_norm, dt, title=f"Total L² Norm Over Time {shape}")



# Mitosis
Du = 0.14
Dv = 0.06
f  = 0.035
k  = 0.065
N_t = 10000
run_full(Du, Dv, f, k, N_t, c_v_init, r, dt,dx,shape="Mitosis")
run_full(Du, Dv, f, k, N_t, c_v_init, r, dt,dx,shape="Mitosis (noise)", add_noise=True)

# Coral Pattern
Du = 0.16
Dv = 0.08
f  = 0.060
k  = 0.062
N_t=10000
run_full(Du, Dv, f, k, N_t, c_v_init, r, dt,dx,shape="Coral")
run_full(Du, Dv, f, k, N_t, c_v_init, r, dt,dx,shape="Coral (noise)", add_noise=True)

# Spirals
N_t=10000
Du, Dv, f, k = 0.12, 0.08, 0.020, 0.050
run_full(Du, Dv, f, k, N_t, c_v_init, r, dt,dx,shape="Spirals")
run_full(Du, Dv, f, k, N_t, c_v_init, r, dt,dx,shape="Spirals (noise)", add_noise=True)

# Zebra fish
Du, Dv, f, k = 0.16, 0.08, 0.035, 0.060
run_full(Du, Dv, f, k, N_t, c_v_init, r, dt,dx,shape="Zebra")
run_full(Du, Dv, f, k, N_t, c_v_init, r, dt,dx,shape="Zebra (noise)", add_noise=True)

