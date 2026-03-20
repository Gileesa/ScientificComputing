import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Grid ---
Lx, Ly = 10.0, 8.0
nx, ny = 200, 160
dx = Lx / nx
h = dx

# --- Time ---
c = 1.0
dt = 0.4 * h / c
nt = 3000

# --- Mesh ---
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# --- Refractive index ---
n = np.ones((nx, ny), dtype=np.complex128)
n_wall = 2.5 + 0.5j



# walls
n[int(2.5/dx):int(2.7/dx), int(2/dx):int(6/dx)] = n_wall
n[int(7.3/dx):int(7.5/dx), int(2/dx):int(6/dx)] = n_wall

alpha = c / n
a = alpha**2  # this is what enters PDE

# --- Gaussian source ---
frequency = 2.4  # GHz (just for reference)
omega = 10.0     # scaled angular frequency (important: not too large!)
A = 10**4
sigma = 0.2
xr, yr = 5.0, 4.0

fxy = A * np.exp(-((X - xr)**2 + (Y - yr)**2)/(2 * sigma**2))

# --- Initialize ---
u_prev = np.zeros((nx, ny), dtype=np.complex128)
u = np.zeros((nx, ny), dtype=np.complex128)

def divergence_term(u, a):
    """Compute ∇·(a ∇u) using flux form"""
    term = np.zeros_like(u)

    # interior only
    # x-direction fluxes
    a_ip = 0.5 * (a[1:-1,1:-1] + a[2:,1:-1])
    a_im = 0.5 * (a[1:-1,1:-1] + a[:-2,1:-1])

    flux_x = a_ip * (u[2:,1:-1] - u[1:-1,1:-1]) - \
             a_im * (u[1:-1,1:-1] - u[:-2,1:-1])

    # y-direction fluxes
    a_jp = 0.5 * (a[1:-1,1:-1] + a[1:-1,2:])
    a_jm = 0.5 * (a[1:-1,1:-1] + a[1:-1,:-2])

    flux_y = a_jp * (u[1:-1,2:] - u[1:-1,1:-1]) - \
             a_jm * (u[1:-1,1:-1] - u[1:-1,:-2])

    term[1:-1,1:-1] = (flux_x + flux_y) / h**2
    return term

# --- First step ---
u_next = u + 0.5 * dt**2 * (divergence_term(u, a) + fxy)

# --- Storage ---
frames = []

# --- Time stepping ---
for t in range(nt):
    div = divergence_term(u, a)

    t_real = t * dt
    source_t = fxy * np.sin(omega * t_real)

    u_new = 2*u - u_prev + dt**2 * (div + source_t)

    # boundaries
    u_new[0,:] = 0
    u_new[-1,:] = 0
    u_new[:,0] = 0
    u_new[:,-1] = 0

    u_prev, u = u, u_new

    if t % 3 == 0:
        frames.append(u.copy())

# --- Plot ---
fig, ax = plt.subplots(figsize=(8,6))

im = ax.imshow(np.real(frames[0]).T, origin='lower',
               extent=[0, Lx, 0, Ly],
               cmap='RdBu')

# walls overlay
wall_mask = np.abs(n - 1) > 1e-6
overlay = np.ma.masked_where(~wall_mask, wall_mask)
ax.imshow(overlay.T, origin='lower',
          extent=[0, Lx, 0, Ly],
          cmap='gray', alpha=0.4)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Wave amplitude (Re[u])")

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                    color='white', fontsize=12,
                    bbox=dict(facecolor='black', alpha=0.7))

ax.set_title("Physically Correct Wave with Variable Refractive Index")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

def animate(i):
    im.set_data(np.real(frames[i]).T)
    time_text.set_text(f"t = {i*dt*3:.3f} s")
    return [im, time_text]

ani = animation.FuncAnimation(fig, animate,
                              frames=len(frames),
                              interval=50)

plt.tight_layout()
plt.show()