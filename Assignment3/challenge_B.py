import numpy as np
import matplotlib.pyplot as plt
import cmath
import os
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

scale = 20  #scaling so that 1 meter is 20 pixels
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
scale_freq = 1/6

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
def initilize_walls(walls, outer_walls, show_plot = False):
    wall_mask = np.zeros((nx,ny), dtype=int)
    for wall in walls:
        place_wall(wall_mask, wall[0], wall[1], wall[2], wall[3], scale, wall_thickness, nx, ny)

    for wall in outer_walls:
        place_wall(wall_mask, wall[0], wall[1], wall[2], wall[3], scale, wall_thickness, nx, ny, outer_wall=True)

    #testing to see if the walls are correctly placed
    if show_plot:
        plt.figure(figsize=(10,9))

        plt.imshow(wall_mask.T, origin='lower', cmap='binary')
        plt.title("Wall layout")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    return wall_mask

def initilize_k_field(wall_mask, nx, ny, frequency):
    c = 3 * 10**8
    k0 = 2 * np.pi * frequency * 10**9 / c
    k = np.full((nx, ny), k0, dtype=np.complex128)
    n_wall = 2.5 + 0.5j
    k[wall_mask==1] = k0 * n_wall
    return k

#this function places the router as a source in coordinates (xr, yr)
def initilize_source(xr, yr, dx, nx, ny, Lx, Ly, A, sigma):
    fxy = np.zeros((nx, ny), dtype=float)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    fxy = A * np.exp(-((X - xr)**2 + (Y - yr)**2)/(2 * sigma**2))
    return fxy



#sparse matrix
def helmholtz(
    k: np.ndarray,
    fxy: np.ndarray,
    nx: int,
    ny: int,
    dx: float,
) -> np.ndarray:
    """
    Function that solves Helmholtz equation on a 2D finite difference grid for wifi simulation.

    Helmholtz equation is as follows:

        Δu + k(x, y)^2 u = f(x, y)

    We solve this on a 2D grid, using a 5-point finite difference scheme. 
    This leads to a linear system A u = b, which we assemble compressed sparse row (CSR) format 
    and solve with a Scipy sparse solver.

    Params:
    - k : np.ndarray, shape (nx, ny)
        Spatially varying wavenumber field, complex-valued. In air we take
        k = k0, in walls k = n_wall * k0.
    - fxy : np.ndarray, shape (nx, ny)
        Source term f(x, y) on the grid (e.g. Gaussian at wifi router location).
    - nx, ny : int
        Number of grid points in x- and y-direction.
    - dx : float
        Grid spacing (we assume dx = dy).

    Returns:
    - u : np.ndarray, shape (nx, ny)
        Complex solution field u(x, y) of the discrete Helmholtz equation.
    """
        
    N = nx * ny

    def idx(i, j):
        """
        Map 2D grid indices (i, j) to 1D index n = i * ny + j
        """
        return i * ny + j

    # Build index arrays for all grid points
    # nn is the flat index
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    nn = idx(ii, jj)

    # Find boundary vs interior points
    is_bnd = (ii == 0) | (ii == nx-1) | (jj == 0) | (jj == ny-1)
    is_int = ~is_bnd

    # We will assemble the matrix in compressed sparse row format; i.e three 1D arrays (rows, cols, vals)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []

    # r.h.s vector of EQ Ax = b
    b = np.zeros(N, dtype=np.complex128)


    # Interior points: finite difference discretisation of Helmholtz
    # Get interior points
    n_int = nn[is_int]
    i_int = ii[is_int]
    j_int = jj[is_int]
    k_int = k[i_int, j_int]

    # Diagonal entries: (k_ij^2 dx^2 - 4) * u_ij
    rows.extend(n_int)
    cols.extend(n_int)
    vals.extend(k_int**2 * dx**2 - 4.0)

    # Off-diagonal entries for the four neighbours: up, down, left, right
    # Each neighbour contributes +1 to the corresponding column.
    # u_U + u_D + u_L + u_R
    for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
        rows.extend(n_int)
        cols.extend(idx(i_int + di, j_int + dj))
        vals.extend(np.ones(len(n_int)))

    # r.h.s for interior points: - f_ij dx^2
    b[n_int] = -fxy[i_int, j_int] * dx**2



    # Boundary points: absorbing BCs
    # Get boundary points
    n_bnd = nn[is_bnd]
    i_bnd = ii[is_bnd]
    j_bnd = jj[is_bnd]
    k_bnd = k[i_bnd, j_bnd]

    # For each boundary point, find the index of the closest interior neighbour
    # in the normal direction, since this is the direction of flow. 
    ni_bnd = np.where(i_bnd == 0,      idx(1,      j_bnd),
             np.where(i_bnd == nx-1,   idx(nx-2,   j_bnd),
             np.where(j_bnd == 0,      idx(i_bnd,  1),
                                       idx(i_bnd,  ny-2))))


    # Coefficient for u_bnd (diagonal)
    rows.extend(n_bnd)
    cols.extend(n_bnd)
    vals.extend(1.0 - 1j * k_bnd * dx)

    # Coefficient for u_inner (neighbour inside the domain)
    rows.extend(n_bnd)
    cols.extend(ni_bnd)
    vals.extend(-np.ones(len(n_bnd), dtype=np.complex128))

    # Right-hand side is zero for boundary equations
    b[n_bnd] = 0.0

    # Assemble CSR matrix
    A = csr_matrix(
        (np.array(vals, dtype=np.complex128), (np.array(rows), np.array(cols))),
        shape=(N, N)
    )

    # Solve
    u_flat = spsolve(A, b)
    print("Solved sparse system")

    # reshape back to normal 2D grid
    u = u_flat.reshape(nx, ny)
    return u


def average_signal_db(u_db):
    # convert dB → linear
    power = 10**(u_db / 10)

    # average
    power_avg = np.mean(power)

    # back to dB
    db_avg = 10 * np.log10(power_avg)

    return db_avg

def compute_wifi_strength(Lx:float,Ly:float, nx:int, ny:int, positions: list[tuple[float, float]], umatrix: np.ndarray, names:list = [], routerpos:tuple[float,float]=None):
    """
    Function that computes wifi strenghts at specific locations
    
    Params:
    - Lx, Ly: float
        Lengths of X and Y dimension in meters
    - nx, ny: int
        number of grid points on x- and y-axis
    - positions: list[tuple[float, float]
        location tuples of (x,y) positions of all measurements
    - umatrix: np.ndarray
        matrix of final wifi strength of simulation. Given in Power (not dB)
    - names: list[str]
        list of strings containing room names
    - routerpos: tuple[float, float]
        (x,y) position of wifi router
    """

    strenghts = []

    print("\n===== WIFI SIGNAL STRENGTH=====")
    for x,y in positions:
        # transform meter to grid position
        # NOTE: we are measuring on a 5x5 cm grid. This is approx the same as a 5cm radius circle
        x_grid = int(x/Lx * nx)
        y_grid = int(y/Ly * ny)
        # strength = np.abs(umatrix[x_grid, y_grid])
        # strength_db = 20 * np.log10(strength  + 10 ** (-12))
        # strenghts.append(strength_db)
        strength = local_average(umatrix, x_grid, y_grid, radius_pixels=1)
        strength_db = 20 * np.log10(strength + 1e-12)
        strenghts.append(strength_db)

    if names:
        for i, name in enumerate(names):
            print(f"{name}: {strenghts[i]:.4f}")
    
    linear_vals = [10**(s/20) for s in strenghts]
    avg_linear = np.mean(linear_vals)
    avg_db = 20 * np.log10(avg_linear + 1e-12)
    print("=> AVERAGE: ", avg_db)

    save_folder = "wifi_results"
    os.makedirs(save_folder, exist_ok=True)

    if routerpos is not None:
        rx, ry = routerpos
        filename = f"wifi_strength_rx{rx}_ry{ry}.csv"
    else:
        filename = "wifi_strength.csv"

    filepath = os.path.join(save_folder, filename)

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)

        # header
        writer.writerow(["Location", "x", "y", "Strength"])

        # data
        for i, (x, y) in enumerate(positions):
            name = names[i] if i < len(names) else f"Point_{i}"
            writer.writerow([name, x, y, strenghts[i]])

    return strenghts



#wrapper function that runs the whole simulation
def run_sim(Lx, Ly, nx, ny, scale, dx, xr, yr, walls, outer_walls, wall_thickness, frequency, scale_freq, A, sigma, max_run):
    freq_scaled = frequency * scale_freq
    wall_mask = initilize_walls(walls, outer_walls, show_plot=True)
    k = initilize_k_field(wall_mask, nx, ny, freq_scaled)
    fxy = initilize_source(xr, yr, dx, nx, ny, Lx, Ly, A, sigma)
    u = helmholtz(k, fxy, nx, ny, dx)

    positions = [(1,5), (2,1), (9,1), (9,7)]
    names = ["Living Room", "Kitchen", "Bathroom", "Bedroom"]
    compute_wifi_strength(Lx, Ly, nx,ny,positions,u,names,(xr,yr))

    plot_sim(u, Lx, Ly, xr, yr, wall_mask, scale, frequency)
    
def plot_sim(u, Lx, Ly, xr, yr, wall_mask, scale, frequency):
    u_abs = np.abs(u)
    u_abs[u_abs == 0] = 1 * 10**-12 #avoiding having 0s in a loglog plot
    u_db = 20 * np.log10(u_abs/np.max(u_abs)) #tried to scale may need changing
    print(np.max(u_db))
    print(np.min(u_db))
    plt.figure(figsize=(8,6))
    #plotting |u| loglog plot
    plt.imshow(u_db.T, origin='lower', extent=[0, Lx, 0, Ly], cmap="jet", vmin=-60, vmax=0)
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
    out_dir = "ScientificComputing/Assignment3/Figures/B/grid_search_results"
    os.makedirs(out_dir, exist_ok=True)

    filename = f"best_router_plot_freq{frequency}_x{xr:.2f}_y{yr:.2f}.pdf"
    filepath = os.path.join(out_dir, filename)

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved best-point plot to {filepath}")
    plt.show()

def local_average(u, xg, yg, radius_pixels=1):
    vals = []
    for i in range(xg - radius_pixels, xg + radius_pixels + 1):
        for j in range(yg - radius_pixels, yg + radius_pixels + 1):
            if 0 <= i < u.shape[0] and 0 <= j < u.shape[1]:
                if (i - xg)**2 + (j - yg)**2 <= radius_pixels**2:
                    vals.append(np.abs(u[i, j]))
    return np.mean(vals)

def evaluate_router_position(xr, yr, Lx, Ly, nx, ny, scale, dx, walls, outer_walls,
                             wall_thickness, frequency, scale_freq, A, sigma):
    freq_scaled = frequency * scale_freq
    wall_mask = initilize_walls(walls, outer_walls, show_plot=False)
    k = initilize_k_field(wall_mask, nx, ny, freq_scaled)
    fxy = initilize_source(xr, yr, dx, nx, ny, Lx, Ly, A, sigma)
    u = helmholtz(k, fxy, nx, ny, dx)

    positions = [(1,5), (2,1), (9,1), (9,7)]
    names = ["Living Room", "Kitchen", "Bathroom", "Bedroom"]

    strengths_db = []
    for x, y in positions:
        xg = min(int(x / Lx * nx), nx - 1)
        yg = min(int(y / Ly * ny), ny - 1)
        # amp = np.abs(u[xg, yg])
        # amp = local_average(u, xg, yg)
        # db = 20 * np.log10(amp + 1e-12)
        amp = local_average(u, xg, yg, radius_pixels=1)
        db = 20 * np.log10(amp + 1e-12)
        strengths_db.append(db)

    # score = sum(strengths_db)
  
    linear_vals = [10**(s/20) for s in strengths_db]
    score = sum(linear_vals)
    # score = min(strengths_db)   # maximize the weakest room
    return score, strengths_db, names

def router_worker(args):
    xr, yr, Lx, Ly, nx, ny, scale, dx, walls, outer_walls, wall_thickness, frequency, scale_freq, A, sigma = args

    freq_scaled = frequency * scale_freq
    wall_mask = initilize_walls(walls, outer_walls, show_plot=False)
    k = initilize_k_field(wall_mask, nx, ny, freq_scaled)
    fxy = initilize_source(xr, yr, dx, nx, ny, Lx, Ly, A, sigma)
    u = helmholtz(k, fxy, nx, ny, dx)

    positions = [(1,5), (2,1), (9,1), (9,7)]
    names = ["Living Room", "Kitchen", "Bathroom", "Bedroom"]

    strengths_db = []
    for x, y in positions:
        xg = min(int(x / Lx * nx), nx - 1)
        yg = min(int(y / Ly * ny), ny - 1)

        # amp = np.abs(u[xg, yg])
        # db = 20 * np.log10(amp + 1e-12)
        # strengths_db.append(db)

        amp = local_average(u, xg, yg, radius_pixels=1)
        db = 20 * np.log10(amp + 1e-12)
        strengths_db.append(db)

    linear_vals = [10**(s/20) for s in strengths_db]
    score = sum(linear_vals)
        
    # score = min(strengths_db)
    return xr, yr, score, strengths_db, names

def grid_search_router_parallel(Lx, Ly, nx, ny, scale, dx, walls, outer_walls,
                                wall_thickness, frequency, scale_freq, A, sigma,
                                x_values, y_values, max_workers=None):
    best_score = -np.inf
    best_pos = None
    best_strengths = None
    results = []

    tasks = []
    for xr in x_values:
        for yr in y_values:
            tasks.append((
                xr, yr, Lx, Ly, nx, ny, scale, dx,
                walls, outer_walls, wall_thickness,
                frequency, scale_freq, A, sigma
            ))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(router_worker, task) for task in tasks]

        for future in as_completed(futures):
            xr, yr, score, strengths_db, names = future.result()
            results.append((xr, yr, score, strengths_db))

            if score > best_score:
                best_score = score
                best_pos = (xr, yr)
                best_strengths = strengths_db

            print(f"tested ({xr:.2f}, {yr:.2f}) -> score {score:.2f} dB")

    save_folder = "ScientificComputing/Assignment3/Figures/B/grid_search_results"
    os.makedirs(save_folder, exist_ok=True)

    filepath = os.path.join(save_folder, "router_grid_search_parallel.csv")
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["xr", "yr", "score (sum dB)", "Living", "Kitchen", "Bathroom", "Bedroom"])
        for xr, yr, score, strengths in results:
            writer.writerow([xr, yr, score] + strengths)

    best_filepath = os.path.join(save_folder, "best_router_position_parallel.csv")
    with open(best_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["xr", "yr", "score"])
        writer.writerow([best_pos[0], best_pos[1], best_score])

    print(f"Saved grid search results to {filepath}")
    print(f"Saved best position to {best_filepath}")

    print("\nBest router position:", best_pos)
    print("Best summed score:", best_score)
    for name, s in zip(names, best_strengths):
        print(f"{name}: {s:.2f} dB")

    return best_pos, best_score, results


# frequency = 2.4
# # xr = 6
# # yr = 3
# xr, yr = best_pos
# run_sim(Lx, Ly, nx, ny, scale, dx, xr, yr, walls, outer_walls, wall_thickness, frequency, scale_freq, A, sigma, 90000)

if __name__ == "__main__":
    frequency = 2.4

    x_values = np.arange(1.0, 9.5, 0.5)
    y_values = np.arange(1.0, 7.5, 0.5)

    best_pos, best_score, results = grid_search_router_parallel(
        Lx, Ly, nx, ny, scale, dx,
        walls, outer_walls, wall_thickness,
        frequency, scale_freq, A, sigma,
        x_values, y_values,
        max_workers=4
    )

    print("Best position found:", best_pos)
    print("Best score:", best_score)

    xr, yr = best_pos
    run_sim(
        Lx, Ly, nx, ny, scale, dx, xr, yr,
        walls, outer_walls, wall_thickness,
        frequency, scale_freq, A, sigma, 90000
    )