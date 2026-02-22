# Wave Equation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl


def initial_wave_profile_cond(u, N, r):
    ''' 
    Function that computes first time step of wave 
    for specific boundary conditions using truncated Taylor expansion.
    Bound. Cond: sin(5*pi*x) if 1/5 < x < 2/5, else 0
    Includes boundary conditions u=0 at x=0 and x=N-1

    Params:
    - u: initial wave, used for iteration
    - N: length of spatial discretisation
    - r: constant (c * dt / dx)
    Returns:
    - u: first iteration of wave profile
    '''
    for i in range(1, N):
        # initial wave profile (displacement at time t=1) -> use Taylor expansion to find u[1, i]
        if x[i] <= (1/5) or x[i] >= (2/5):
            u[1, i] = 0
        else:
            u[1, i] = u[0, i] + 0.5 * r**2 * (u[0, i+1] - 2*u[0, i] + u[0, i-1])
    u[1, 0] = 0.0
    u[1, -1] = 0.0
    return u


def propagate_wave_cond(u, N, Nt, x, r):
    '''
    Function that propagates wave using iteration method 
    for wave: sin(5*pi*x) if 1/5 < x < 2/5, else 0
    Includes boundary conditions u=0 at x=0 and x=N-1

    Params:
    - u: initial wave, used for iteration
    - N: length of spatial discretisation
    - Nt: number the time steps
    - r: constant (c * dt / dx)
    Returns:
    - u: first iteration of wave profile

    '''
    for n in range(1, Nt):
        for i in range(1, N):
            if x[i] <= (1/5) or x[i] >= (2/5):
                u[1, i] = 0
            else:
                u[n+1, i] = 2*u[n, i] - u[n-1, i] + r**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
        u[n+1, 0] = 0.0
        u[n+1, -1] = 0.0
    return u 



def initial_wave_profile(u, N, r):
    ''' 
    Function that computes first time step of wave 
    for specific boundary conditions using truncated Taylor expansion.
    Includes boundary conditions u=0 at x=0 and x=N-1

    Params:
    - u: initial wave, used for iteration
    - N: length of spatial discretisation
    - r: constant (c * dt / dx)
    Returns:
    - u: first iteration of wave profile
    '''
    for i in range(1, N):
        # initial wave profile (displacement at time t=1) -> use Taylor expansion to find u[1, i]
        u[1, i] = u[0, i] + 0.5 * r**2 * (u[0, i+1] - 2*u[0, i] + u[0, i-1])
    u[1, 0] = 0.0
    u[1, -1] = 0.0
    return u

def propagate_wave(u, N, Nt, r):
    '''
    Function that propagates wave using iteration method 
    for wave: sin(5*pi*x) if 1/5 < x < 2/5, else 0
    Includes boundary conditions u=0 at x=0 and x=N-1

    Params:
    - u: initial wave, used for iteration
    - N: length of spatial discretisation
    - Nt: number the time steps
    - r: constant (c * dt / dx)
    Returns:
    - u: first iteration of wave profile

    '''
    for n in range(1, Nt):
        for i in range(1, N):
            u[n+1, i] = 2*u[n, i] - u[n-1, i] + r**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
        u[n+1, 0] = 0.0
        u[n+1, -1] = 0.0
    return u 


# Question C
def animate_wave(u_matrix, filename, dt, T, equation=""):
    '''
    Function that animates a wave in 2D
    Assumes each row in the u_matrix is one time step

    Params:
    - u_matrix: matrix containing wave over time. Each row represents one time step
    '''
    N_t, N_x = u_matrix.shape
    x = np.linspace(0, 1, N_x)

    step = 5
    frame_indices = np.arange(0, N_t, step)

    fig, ax = plt.subplots()
    line, = ax.plot(x, u_matrix[0])

    # set axis labels
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("Amplitude")
    ax.set_title("Wave Animation of " + filename + f" ({equation})")

    # set limits for proper animation
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(u_matrix.min(), u_matrix.max())

    # Colormap
    cmap = plt.get_cmap('plasma')

    # Colorbar (shows time)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=T))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=plt.gca())
    cbar.set_label('Time')

    def update(frame_idx):
        t = frame_idx * dt
        line.set_ydata(u_matrix[frame_idx]) # make animation faster by *5\
        line.set_color(cmap(t))
        return line,

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=50,
        blit=True
    )
    if equation == "":
        ani.save(filename=filename + ".gif", writer="pillow", fps=30)
    else: 
        ani.save(filename=filename + "_boundaries.gif", writer="pillow", fps=30)
    plt.show()

# Question B
def plot_wave(u, eq, Nt, dt, x, filename):
    cmap = plt.get_cmap('plasma', Nt)
    for i in range(0, Nt):
        t = i * dt
        colour = cmap(i)
        plt.plot(x, u[i, :], color = colour)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=T))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Time')

    plt.xlabel(r"$x$")
    plt.ylabel(f"Amplitude")
    plt.title(f"Wave evolution u(x,t)={eq}")
    plt.savefig(filename + ".png")
    plt.show()

def run_1_1():
    '''
    Function that runs full wave equation pipeline.

    '''
    
    L = 1 # length of the string
    c = 1 # wave speed (m/s)

    N = 1000 # number of spatial points
    dx = L / N # spatial step size
    x = np.linspace(0, L, N + 1)
    T = 1  # total time of simulation
    dt = 0.001         # c * (dt / dx) <= 1 for stability
    Nt = int(T / dt) # number of time steps
    r = c * (dt / dx) 

    u = np.zeros((Nt+1, N+1)) # u[n, i] n = time, i = space
    v = np.zeros((Nt+1, N+1))
    y = np.zeros((Nt+1, N+1))

    u[0, :] = np.sin(2*np.pi*x)
    # initial conditions (fixed boundary conditions)
    u[0, 0] = 0.0
    u[0, -1] = 0.0

    v[0, :] = np.sin(5*np.pi*x)
    # initial conditions (fixed boundary conditions)
    v[0, 0] = 0.0
    v[0, -1] = 0.0

    y[0, :] = np.where((x > (1/5)) & (x < (2/5)), np.sin(5 * np.pi * x), 0.0)
    # initial conditions (fixed boundary conditions)
    y[0, 0] = 0.0
    y[0, -1] = 0.0

    # create waves for different init cond.
    u = initial_wave_profile(u, N, r)
    u = propagate_wave(u, N, Nt, r)
    v = initial_wave_profile(v, N, r)
    v = propagate_wave(v, N, Nt, r)
    y = initial_wave_profile(y, N, r)
    y = propagate_wave(y, N, Nt, r)

    # Question B
    plot_wave(u, 'sin(2πx)', Nt, dt, x, "wave_sin(2πx)")
    plot_wave(v, 'sin(5πx)', Nt, dt, x, "wave_sin(5πx)")
    plot_wave(y, 'sin(5πx), if 1/5 < x < 2/5 else Ψ = 0', Nt, dt, x, "wave_sin(5πx)_boundaries")

    # Question C
    animate_wave(u, "sin(2πx)", dt, T)
    animate_wave(v, "sin(5πx)", dt, T)
    animate_wave(y, "sin(5πx)", dt, T, equation="if 1/5 < x < 2/5 else Ψ = 0")

    print("all saved")
    print("1.1 finished running")

run_1_1()