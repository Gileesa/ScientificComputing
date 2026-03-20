import numpy as np
import matplotlib.pyplot as plt
import cmath
import os
import matplotlib.animation as animation

# Create discretisation grid
scale = 40  #scaling so that 1 meter is 20 pixels
dx = 1/scale    #each pixel is 1/20 cm

Lx, Ly = 10.0, 8.0  #meters
nx, ny = int(Lx * scale), int(Ly * scale) #scaling the outer walls

t = int(round(0.15 * scale)) #needs to be an odd number, if you change this you need to make other changes otherwise the wall code breaks
#since grid[x,y] when plotting make sure to do plt.imshow(grid.T, origin='lower') or plt.imshow(np.flipud(grid.T)) 
wall_thickness = 2*(t//2)+1
    
#wall coordinates [wall1[x1, y1, x2, y2], wall2[x1,y1,x2,y2]]
walls = [[2.5, 0, 2.5, 2], [7, 0, 7, 1.5], [7, 2.5, 7, 3], [6, 3, 6, 8], #vertical
         [0, 3, 3, 3], [4, 3, 6, 3], [7, 3, 10, 3]] #horizontal
outer_walls = [[0, 0, 0, 8], [0, 0, 10, 0], [10, 0, 10, 8], [0, 8, 10, 8]]

#scale for the freq like done in the example given
scale_freq = 1/3

#guassian pulse variables
A = 1 * 10**4
sigma = 0.2

#helper functions
#this function helps to scale each x and y point
def scaling(x, y, scale):
    return int(x * scale), int(y * scale)

#this function is used for placing the walls
def place_wall(wall_mask, x1, y1, x2, y2, scale, wall_thickness, nx, ny, outer_wall = False):
    half_thickness = wall_thickness // 2
    i1, j1 = int(x1 * scale), int(y1 * scale)
    i2, j2 = int(x2 * scale), int(y2 * scale)
    if i1 == i2: #vertical wall
        x = i1
        #this is to ensure that if we are on the edge we do not go negative
        xmin = max(0, x - half_thickness) 
        xmax = min(nx, x + half_thickness + 1)
        if outer_wall and x == 0:
            wall_mask[x:wall_thickness, :] = 1
            return
        elif outer_wall and x == nx:
            wall_mask[x-wall_thickness:x, :] = 1
            return
        wall_mask[xmin:xmax, min(j1,j2):max(j1,j2)] = 1
    if j1 == j2: #horizontal wall
        y = j1
        ymin = max(0, y - half_thickness)
        ymax = min(ny, y + half_thickness + 1)
        if outer_wall and y == 0:
            wall_mask[:, 0:wall_thickness] = 1
            return
        elif outer_wall and y == ny:
            wall_mask[:, y-wall_thickness:y] = 1
            return
        wall_mask[min(i1,i2):max(i1,i2), ymin:ymax] = 1

#initilizer functions
#this function given wall coordinates places walls according to it, you need to give inner and outer walls seperately
def initilize_walls(walls, outer_walls):
    wall_mask = np.zeros((nx,ny), dtype=int)
    for wall in walls:
        place_wall(wall_mask, wall[0], wall[1], wall[2], wall[3], scale, wall_thickness, nx, ny)

    for wall in outer_walls:
        place_wall(wall_mask, wall[0], wall[1], wall[2], wall[3], scale, wall_thickness, nx, ny, outer_wall=True)

    #testing to see if the walls are correctly placed
    plt.figure(figsize=(10,9))

    plt.imshow(wall_mask.T, origin='lower', cmap='binary')
    plt.title("Wall layout")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    return wall_mask

def initilize_k_field(wall_mask, nx, ny, frequency):
    c = 3 * 10**8 # speed of light
    k0 = 2 * np.pi * frequency / c
    k = np.full((nx, ny), k0, dtype=np.complex128)
    n_wall = 2.5 + 0.5j
    k[wall_mask==1] = k0 * n_wall
    return k

# this function places the router as a source in coordinates (xr, yr)
def initilize_source(xr, yr, dx, nx, ny, Lx, Ly, A, sigma):
    fxy = np.zeros((nx, ny), dtype=float)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    fxy = A * np.exp(-((X - xr)**2 + (Y - yr)**2)/(2 * sigma**2))
    return fxy

# main loop helmholtz run
def helmholtz(k, fxy, nx, ny, dx, wall_mask, max_run = 50000):
    u = np.zeros((nx, ny), dtype=np.complex128)
    u_neighbourhood = np.zeros((nx, ny), dtype=np.complex128)
    omega = 0.6 # for SOR

    for iter in range(max_run):
        u_old = u.copy()
        denom = 4.0 - k**2 * dx**2
        mask = np.abs(denom) < 1e-12
        denom[mask] = 1e-6 * denom[mask]/np.abs(denom[mask] + 1e-12)

        u_neighbourhood[1:-1, 1:-1] = ((
            u_old[1:-1, 2:]   +  # right
            u_old[1:-1, :-2]  +  # left
            u_old[2:, 1:-1]   +  # up
            u_old[:-2, 1:-1]  +  # down
            - fxy[1:-1, 1:-1] * dx**2) 
            / denom[1:-1, 1:-1])
        
        u = (1 - omega)*u + omega * u_neighbourhood

        #absorbing boundary conditions where it depends on its closest non boundary neighbour
        for l in range(ny):
            #left wall
            u[0, l] = u[1, l] / (1 - 1j * k[0, l] * dx)
            #right wall
            u[-1, l] = u[-2, l] / (1 - 1j * k[-1, l] * dx)
        for i in range(nx):
            #top wall
            u[i, 0] = u[i, 1] / (1 - 1j * k[i, 0] * dx)
            #bottom wall
            u[i, -1] = u[i, -2] / (1 - 1j * k[i, -1] * dx)

        conv = np.max(np.abs(u - u_old))
        num = max(np.max(np.abs(u)), 1e-12)
        if iter % 500 == 0:
            print(f"num: {num}")
            print(f"rel change: {conv/num}")
        if conv/num < 1 * 10**-5: #changing the tolerance to something lower makes it not converge
            print(f"system converged at iteration {iter}")
            return u
    print(f"system failed to converge in {max_run} steps")
    return u


def animate_wave(u, Lx, Ly, xr, yr, wall_mask, frequency, num_frames=60, save_file=None):
    """
    Animate the time-varying wave field from Helmholtz solution.
    
    Parameters:
    - u: Complex wave field from helmholtz solver
    - frequency: Frequency in GHz
    - num_frames: Number of frames in animation
    - save_file: If provided, save animation to this file (e.g., 'wave.mp4')
    """
    
    omega = 2 * np.pi * frequency * 1e9  # Angular frequency (rad/s)
    magnitude = np.abs(u)
    # magnitude = u
    phase = np.angle(u)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initial frame
    wave_real = magnitude * np.cos(phase)
    im = ax.imshow(wave_real.T, origin='lower', extent=[0, Lx, 0, Ly],
                   cmap='RdBu', vmin=-magnitude.max(), vmax=magnitude.max())
    
    # Overlay walls
    wall_alpha = np.ma.masked_where(wall_mask == 0, wall_mask)
    ax.imshow(wall_alpha.T, origin='lower', extent=[0, Lx, 0, Ly], 
              cmap='gray', alpha=0.3)
    
    # Router position
    ax.scatter(xr, yr, marker='*', s=200, c='yellow', edgecolors='black', 
               linewidths=2, label='Router', zorder=10)
    
    # Time display
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                        color='white', fontsize=14, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    ax.set_title(f'WiFi Wave Oscillation at {frequency} GHz', fontsize=16)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Wave Amplitude')
    ax.legend(loc='upper right')
    
    def animate_frame(frame):
        # One full period
        t = frame / num_frames * (1 / (frequency * 1e9))
        
        # Real part of wave: Re[u * exp(iωt)]
        wave_real = magnitude * np.cos(omega * t + phase)
        
        im.set_data(wave_real.T)
        time_text.set_text(f't = {t*1e12:.2f} ps')  # Picoseconds
        
        return [im, time_text]
    
    anim = animation.FuncAnimation(fig, animate_frame, frames=num_frames, 
                                   interval=50, blit=True)
    
    if save_file:
        anim.save(save_file, writer='pillow', fps=20)
        print(f"Animation saved to {save_file}")
    
    plt.tight_layout()
    plt.show()
    
    return anim

def animate_wave_complex(u, Lx, Ly, xr, yr, wall_mask, frequency, num_frames=60):
    omega = 2 * np.pi * frequency * 1e9  

    magnitude = np.abs(u)
    phase = np.angle(u)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Initial frame
    wave_real = magnitude * np.cos(phase)

    im = ax.imshow(wave_real.T,
                   origin='lower',
                   extent=[0, Lx, 0, Ly],
                   cmap='RdBu',
                   vmin=-magnitude.max(),
                   vmax=magnitude.max())

    # --- WALL OVERLAY ---
    wall_overlay = np.ma.masked_where(wall_mask == 0, wall_mask)
    ax.imshow(wall_overlay.T,
            origin='lower',
            extent=[0, Lx, 0, Ly],
            cmap='gray',
            alpha=0.4,
            zorder=11)

    # --- ROUTER MARKER ---
    ax.scatter(xr, yr,
            marker='*',
            s=200,
            c='yellow',
            edgecolors='black',
            linewidths=1.5,
            label='Router',
            zorder=10)

    # --- COLORBAR ---
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Wave amplitude (Re[u])")

    # --- TIME LABEL ---
    time_text = ax.text(0.02, 0.95, '',
                        transform=ax.transAxes,
                        color='white',
                        fontsize=12,
                        bbox=dict(facecolor='black', alpha=0.7))

    # --- AXIS LABELS ---
    ax.set_title(f"Wave Field (Helmholtz Solution)\nFrequency = {frequency} GHz")
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")

    ax.legend(loc='upper right')

    # --- ANIMATION FUNCTION ---
    def animate_frame(frame):
        t = frame / num_frames * (2*np.pi / omega)

        wave_real = magnitude * np.cos(omega * t + phase)
        im.set_data(wave_real.T)

        time_text.set_text(f"t = {t*1e12:.2f} ps")  # picoseconds

        return [im, time_text]

    anim = animation.FuncAnimation(fig,
                                   animate_frame,
                                   frames=num_frames,
                                   interval=50,
                                   blit=True)

    plt.tight_layout()
    plt.show()

    return anim

# wrapper function that runs the whole simulation
def run_sim(Lx, Ly, nx, ny, scale, dx, xr, yr, walls, outer_walls, wall_thickness, frequency, scale_freq, A, sigma, max_run,  animate=True):
    freq_scaled = frequency * scale_freq
    wall_mask = initilize_walls(walls, outer_walls)
    k = initilize_k_field(wall_mask, nx, ny, freq_scaled)
    fxy = initilize_source(xr, yr, dx, nx, ny, Lx, Ly, A, sigma)
    u = helmholtz(k, fxy, nx, ny, dx, wall_mask, max_run=max_run)

    # static plot
    plot_sim(u, Lx, Ly, xr, yr, wall_mask, scale, frequency)

    # Animated plot (optional) CURRENTLY NOT SAVED
    if animate:
        # animate_wave(u, Lx, Ly, xr, yr, wall_mask, freq_scaled, 
        #              num_frames=60)
        animate_wave_complex(u, Lx, Ly, xr, yr, wall_mask, freq_scaled, num_frames=60)
    
def plot_sim(u, Lx, Ly, xr, yr, wall_mask, scale, frequency):
    u_abs = np.abs(u)
    u_abs[u_abs == 0] = 1 * 10**-12 #avoiding having 0s in a loglog plot
    u_db = 10 * np.log10(u_abs/u_abs.max()) #tried to scale may need changing
    print(np.max(u_db))
    print(np.min(u_db))
    plt.figure(figsize=(8,6))
    #plotting |u| loglog plot
    plt.imshow(u_db.T, origin='lower', extent=[0, Lx, 0, Ly], cmap="jet", vmin=u_db.max(), vmax=u_db.min())
    plt.colorbar(label="Signal Strength (dB)")
    #plotting the walls as an overlay
    wall_alpha = np.ma.masked_where(wall_mask == 0, wall_mask)
    plt.imshow(wall_alpha.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='gray', alpha=1.0)
    #plotting the router position
    plt.scatter(xr, yr, marker='*', s=150, edgecolors='black', label='WiFi Router')
    #adding room labels
    plt.text(2, 6, "Living Room", color='white', bbox=dict(facecolor='grey', alpha=0.6), ha='center')
    plt.text(7.5, 5.5, "Bedroom 1", color='white', bbox=dict(facecolor='grey', alpha=0.6), ha='center')
    plt.text(8.5, 1.5, "Bathroom", color='white', bbox=dict(facecolor='grey', alpha=0.6), ha='center')
    plt.text(5, 1.5, "Hall", color='white', bbox=dict(facecolor='grey', alpha=0.6), ha='center')
    plt.text(1, 1.5, "Kitchen", color='white', bbox=dict(facecolor='grey', alpha=0.6), ha='center')
    #labels
    plt.title(f"WiFi Coverage at {frequency}\nRouter Position: ({xr}, {yr}) m")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    out_dir = "Figures/B"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/freq{frequency}xr{xr}yr{yr}.png")
    plt.show()
frequency = 2.4
xr = 2.5
yr = 5.5
run_sim(Lx, Ly, nx, ny, scale, dx, xr, yr, walls, outer_walls, wall_thickness, frequency, scale_freq, A, sigma, 90000)


