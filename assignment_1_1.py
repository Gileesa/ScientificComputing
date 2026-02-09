# Wave Equation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl


L = 1 # length of the string
c = 1 # wave speed (m/s)

N = 100 # number of spatial points
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

def initial_wave_profile(u, N, r):
    for i in range(1, N):
        # initial wave profile (displacement at time t=1) -> use Taylor expansion to find u[1, i]
        u[1, i] = u[0, i] + 0.5 * r**2 * (u[0, i+1] - 2*u[0, i] + u[0, i-1])
    u[1, 0] = 0.0
    u[1, -1] = 0.0
    return u

def propagate_wave(u, N, Nt, r):
    for n in range(1, Nt):
        for i in range(1, N):
            u[n+1, i] = 2*u[n, i] - u[n-1, i] + r**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
        u[n+1, 0] = 0.0
        u[n+1, -1] = 0.0
    return u 

u = initial_wave_profile(u, N, r)
u = propagate_wave(u, N, Nt, r)
v = initial_wave_profile(v, N, r)
v = propagate_wave(v, N, Nt, r)
y = initial_wave_profile(y, N, r)
y = propagate_wave(y, N, Nt, r)

def animate_wave(u_matrix):
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

    # set limits for proper animation
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(u_matrix.min(), u_matrix.max())

    # Colormap
    cmap = plt.get_cmap('plasma')

    # Colorbar (shows time)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=T))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=plt.gca())
    cbar.set_label('Time (s)')

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
    plt.show()

def plot_wave(u, eq):
    cmap = plt.get_cmap('plasma', Nt)
    for i in range(0, Nt):
        t = i * dt
        colour = cmap(i)
        plt.plot(x, u[i, :], color = colour)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=T))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Time (s)')

    plt.xlabel("x")
    plt.ylabel(f"u(x,t) {eq}")
    plt.title("Wave evolution")
    plt.show()

# plot initial, mid, final
plt.plot(x, u[0, :], label="t = 0")
plt.plot(x, u[1, :], label=f"t = 0.001")
plt.plot(x, u[2, :], label=f"t = 0.002")
plt.legend()
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Wave evolution")
plt.show()

plot_wave(u, 'sin(2πx)')
plot_wave(v, 'sin(5πx)')
plot_wave(y, 'sin(5πx), if 1/5 < x < 2/5 else Ψ = 0')

animate_wave(u)
animate_wave(v)
animate_wave(y)


