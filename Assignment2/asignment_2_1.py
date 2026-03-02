import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from matplotlib.animation import FuncAnimation

bottom = 0
top = 1
grid_size = 100
time_steps = 1000

neighbourhood = [(-1, 0), (1, 0), (0, -1), (0, 1)]
probability_sticking = 0.5

# concentration field
c = np.zeros((grid_size, grid_size), dtype=float)


def diffusion_limited_aggregation(cluster = True, grid_size = 100, time_steps = 1000):
    """
    This function simulates diffusion-limited aggregation (DLA) in a 2D grid. 
    It starts with a seed particle at the center of the grid and randomly releases particles from the edges.
    The particles perform a random walk until they either stick to the existing cluster or move out of bounds. 
    The process continues until a specified number of particles have been added to the cluster.
    param: 
    N: grid_size: The size of the 2D grid (N = 100 for a 100x100 grid).
    t: time steps: The number of particles to be added to the cluster (t = 1000 for 1000 particles).

    Returns: A 2D array representing the final cluster formed by the DLA process.
    
    """
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

    print(c[0, 50])        # should be 0
    print(c[-1, 50])       # should be 1
    print(c[grid_size//2, 50])  # about 0.5
    print(np.max(np.abs(c[:,0] - c[:,1])))     # should be 0 (Neumann)
    print(np.max(np.abs(c[:,-1] - c[:,-2])))   # should be 0 (Neumann)


diffusion_limited_aggregation = diffusion_limited_aggregation()