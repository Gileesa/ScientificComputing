import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from matplotlib.animation import FuncAnimation


neighbourhood = [(-1, 0), (1, 0), (0, -1), (0, 1)]
probability_sticking = 0.5

def diffusion_limited_aggregation(grid_size = 100, bottom = 0, top = 1, time_steps = 1000):
    """
    This function simulates diffusion-limited aggregation (DLA) in a 2D grid. 
    It starts with a seed particle at the center of the grid and randomly releases particles from the edges.
    The particles perform a random walk until they either stick to the existing cluster or move out of bounds. 
    The process continues until a specified number of particles have been added to the cluster.
    param: 
    grid_size: The size of the 2D grid (N = 100 for a 100x100 grid).
    bottom: The value at the bottom boundary (default is 0).
    top: The value at the top boundary (default is 1).
    time_steps: The number of particles to be added to the cluster (t = 1000 for 1000 particles).

    Returns: A 2D array representing the final cluster formed by the DLA process.
    """
    cluster = np.zeros((grid_size, grid_size), dtype=bool)
    cluster[grid_size // 2, grid_size // 2] = True

    obj_matrix = cluster.astype(int)

    # concentration field
    c = np.zeros((grid_size, grid_size), dtype=float)


    # Initialize the cluster with a seed particle at the center of the grid
    for j in range(grid_size):
        value = bottom + (top - bottom) * j / (grid_size - 1)
        c[j, :]  = value
    
    # Neuman boundary conditions &  Dirichlet boundary conditions
    # Set the boundary conditions for the grid. The bottom row is set to 0, and the top row is set to 1.
    # The left and right columns are set to the values of their adjacent columns to create a Neumann boundary condition, which allows for a zero-flux boundary.
    c[0, : ] = bottom 
    c[-1, : ] = top
    c[: , 0] = c[:, 1]
    c[: , -1] = c[:, -2]

    # absorbing boundary conditions for the cluster
    c[obj_matrix == 1] = 0.0

    return c, cluster, obj_matrix


def sor_iteration(c, omega, max_iteration, obj_matrix, epsilon = 10**(-5), bottom = 0, top = 1, save_snap = False):
    """
    This function solves a 2D Laplace equation using the Successive Over-Relaxation (SOR) method.
    We have the following boundary conditions:
    - The top boundary is set to 1 (Dirichlet boundary condition).
    - The bottom boundary is set to 0 (Dirichlet boundary condition).
    - The left and right boundaries are set to the values of their adjacent columns (Neumann boundary condition).

    The function iteratively updates the values in the interior of the matrix until convergence is achieved, 
    which is determined by the maximum change in values being less than a specified epsilon. 
    
    The function also allows for saving snapshots of the concentration field at each iteration if desired.
    
    Parameters:
    - c: (Ny, Nx) ndarray of float. Initial guess for the concentration field.
    - omega: float.The relaxation factor for the SOR method (0 < omega < 2).
    - max_iteration: int. The maximum number of iterations to perform.
    - obj_matrix: A binary matrix indicating the presence of objects (1 for object, 0 for free space) that should be treated as absorbing boundaries.
    - epsilon: float. The convergence criterion (default is 10^(-5)).
    - bottom: float. The value at the bottom boundary (default is 0).
    - top: float. The value at the top boundary (default is 1).
    - save_snap: bool. Whether to save snapshots of the concentration field at each iteration (default is False).

    Returns:
    c : (Ny, Nx) ndarray. Concentration field after convergence (or after max_iteration).
    delta_list : (k,) ndarray. Convergence history: delta[k] = max(|c - c_old|) at each loop.
    k_used : int. Number of loops performed.
    c_over_time : (k, Ny, Nx) ndarray. Saved snapshots if save_snap=True, else an empty array.
    """
    Ny, Nx = c.shape

    delta_list = []
    c_over_time = []

    for iteration in range(max_iteration):
        c_old = c.copy()

        # update the values for in the interior of the matrix, excluding the boundaries
        for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                if obj_matrix[j, i] == 1:
                    c[j, i] = 0
                    continue
            
                neighbour = 0.25 * (c[j - 1, i] + c[j + 1, i] + c[j, i - 1] + c[j, i + 1])
                c[j, i] = (1 - omega) * c[j, i] + omega * neighbour
            
        c[0, : ] = bottom
        c[-1, : ] = top
        c[: , 0] = c[:, 1]
        c[: , -1] = c[:, -2]
        c[obj_matrix == 1] = 0


        delta = np.max(np.abs(c - c_old)) 
        delta_list.append(delta)

        if save_snap:
            c_over_time.append(c.copy())

        if delta < epsilon:
            print(f"SOR omega={omega}, board size={Ny} converged after {iteration + 1} iterations.")
            return c, np.array(delta_list), (iteration + 1), np.array(c_over_time)

    print("WARNING: reached max_iteration without convergence.")
    return c, np.array(delta_list), (k + 1), np.array(c_over_time)



c, cluster, obj_matrix = diffusion_limited_aggregation()

c_solved, deltas, k_used, _ = sor_iteration(
    c, 
    omega = 1.75, 
    max_iteration = 1000, 
    obj_matrix = obj_matrix)

plt.imshow(c_solved, origin="lower")
plt.colorbar()
plt.title("Concentration field after SOR convergence")
plt.show()