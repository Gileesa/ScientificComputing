# Wave Equation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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

u[0, :] = np.sin(2*np.pi*x)
# initial conditions (fixed boundary conditions)
u[0, 0] = 0.0
u[0, -1] = 0.0

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

# def initialise_wave(matrix):
#     Nx = matrix.shape[1]
#     x = np.linspace(0, L, N)
#     matrix[0, :] = np.sin(2 * np.pi * x)
#     # print(matrix[0])
#     return matrix

# A = initialise_wave(A)

# def propagate_wave(A):
#     for j in range(1,t-1):
#         print(' j', j)
#         print(A[j])
#         for i in range(0,N-1):
#             print(i)
#             print(A[j,i])
#             new_u = c**2 * ((dt**2)/(l**2)) * (A[i+1,j] + A[i-1,j] - 2*A[i,j]) - A[i,j-1] + 2*A[i,j]
#             A[i,j+1] = new_u
#         A[0, j+1] = 0
#         A[N-1, j+1] = 0
#     print(A)

# propagate_wave(A)

def animate_wave(u_matrix):
    '''
    Function that animates a wave in 2D
    Assumes each row in the u_matrix is one time step

    Params:
    - u_matrix: matrix containing wave over time. Each row represents one time step
    '''
    N_t, N_x = u_matrix.shape
    x = np.linspace(0, 1, N_x)

    fig, ax = plt.subplots()
    line, = ax.plot(x, u_matrix[0])

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(u_matrix.min(), u_matrix.max())

    def update(t):
        line.set_ydata(u_matrix[t*5]) #make animation faster
        return line,

    ani = FuncAnimation(
        fig,
        update,
        frames=N_t,
        interval=50,
        blit=True
    )
    plt.show()


# plot initial, mid, final
plt.plot(x, u[0, :], label="t = 0")
plt.plot(x, u[Nt//2, :], label=f"t = {Nt//2 * dt:.3f}")
plt.plot(x, u[-1, :], label=f"t = {T}")
plt.legend()
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Wave evolution")
plt.show()

animate_wave(u)


