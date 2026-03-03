import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import os
import concurrent.futures

def initialize_dla_laplace(grid_size = 100, bottom = 0, top = 1, time_steps = 1000):
    """
    This function initializes the state for Laplace growth in a diffusion-limited aggregation (DLA) process. 
    It sets up a 2D grid with specified boundary conditions and initializes the concentration field and cluster.
    param: 
    grid_size: The size of the 2D grid (N = 100 for a 100x100 grid).
    bottom: The value at the bottom boundary (default is 0).
    top: The value at the top boundary (default is 1).
    time_steps: The number of particles to be added to the cluster (t = 1000 for 1000 particles).

    Returns: A 2D array representing the final cluster formed by the DLA process.
    """
    cluster = np.zeros((grid_size, grid_size), dtype=bool)
    cluster[0, grid_size // 2] = True

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


def sor_iteration(c, omega, max_iteration, obj_matrix, epsilon = 10**(-5), bottom = 0, top = 1, save_snap = False, verbose = False):
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
        c = np.clip(c, 0, 1)

        delta = np.max(np.abs(c - c_old)) 
        delta_list.append(delta)

        if save_snap:
            c_over_time.append(c.copy())
        
        if delta < epsilon:
            if verbose:
                print(f"SOR omega={omega}, board size={Ny} converged after {iteration + 1} iterations.")
            return c, np.array(delta_list), (iteration + 1), np.array(c_over_time)

    print("WARNING: reached max_iteration without convergence.")
    return c, np.array(delta_list), (iteration + 1), np.array(c_over_time)


def find_candidates(cluster, neighbourhood = [(-1, 0), (1, 0), (0, -1), (0, 1)]):
    """
    This function identifies the candidate positions for growth in a diffusion-limited aggregation (DLA) cluster.
    It checks the neighboring positions of the existing cluster and returns those that are adjacent to the cluster but not part of it.

    Parameters:
    - cluster: A 2D boolean array representing the current state of the DLA cluster, where True indicates the presence of a particle and False indicates empty space.

    Returns:
    - candidates: A list of tuples, where each tuple contains the (row, column) indices of a candidate position for growth.
    """
    candidates = set()  # avoid duplicates
    for j in range(cluster.shape[0]):
        for i in range(cluster.shape[1]):
            if cluster[j, i]:  # If there is a particle at this position
                for dj, di in neighbourhood:  # Check its neighbors
                    nj, ni = j + dj, i + di
                    if 0 <= nj < cluster.shape[0] and 0 <= ni < cluster.shape[1]:  # Check bounds
                        if not cluster[nj, ni]:  # If the neighbor is empty
                            candidates.add((nj, ni))  # Add to candidates
    return candidates


def growth(cluster, obj_matrix, eta, c, candidates):
    """
    The function chooses the side to grow, which has a higher concentration value, with a probability proportional to the concentration values of the candidate positions.
   
    Parameters:
    - cluster: A 2D boolean array representing the current state of the DLA cluster, where True indicates the presence of a particle and False indicates empty space.
    - obj_matrix: A 2D integer array representing the presence of objects in the grid, where 1 indicates the presence of an object and 0 indicates free space. This matrix is used to update the concentration field after growth.
    - eta: float. A parameter that controls the influence of the concentration values on the growth probability.
    - c: numpy array. A 2D array representing the concentration field.    
    - candidates: A list of tuples, where each tuple contains the (row, column) indices of a candidate position for growth.

    Returns:
    - chosen_position: A tuple containing the (row, column) indices of the chosen position for growth.
    """
    candidate_list = np.array(list(candidates))

    # Get the concentration values at the candidate positions
    candidate_concentrations = []

    for (j, i) in candidate_list:
        concentration = c[j, i]
        if concentration < 0:
            print(f"WARNING: Negative concentration value {concentration} at position ({j}, {i}). Setting it to zero.")
            concentration = 0.0
        candidate_concentrations.append(concentration)

    # Convert the concentration values into weights 
    weights = []

    if eta == 0:
        weights = [1] * len(candidate_concentrations)  # Equal weights for all candidates
    else:
        for conc in candidate_concentrations:
            weights.append(conc ** eta)
    weights_sum = sum(weights)

    # Compute the probabilities for each candidate position
    probabilities = []

    if weights_sum == 0:
        print("WARNING: All candidate concentrations are zero.")
    else:
        for weight in weights:
            probabilities.append(weight / weights_sum)

    # Choose a candidate position based on the computed probabilities
    chosen_index = np.random.choice(len(candidate_list), p=probabilities)
    chosen_position = tuple(candidate_list[chosen_index])

    cluster[chosen_position] = True  # Update the cluster with the new growth

    obj_matrix[chosen_position] = 1  # Update the object matrix to reflect the new growth

    return chosen_position, probabilities, candidate_list, candidate_concentrations, obj_matrix

def diffusion_limited_aggregation(steps = 1000, grid_size = 100, bottom = 0, top = 1, omega = 1.75, eta = 1.0, max_sor_iterations = 1000, save_snap = False, seed = None, save_every_step=True, progress_every=50):
    """
    This fumction simulates a Laplace growth DLA for a given number of growth steps.
    At each step, 
        1. we solve the Laplace equation using the SOR method to find the concentration field,
        2. identify candidate positions for growth, and then 
        3. choose a position to grow based on the concentration values at the candidate positions.
        4. Update the cluster and object matrix accordingly.
    
    Parameters:
    - steps: int. The number of growth steps to simulate (default is 1000).
    - grid_size: int. The size of the 2D grid (default is 100
    - bottom: float. The value at the bottom boundary (default is 0).
    - top: float. The value at the top boundary (default is 1).
    - omega: float. The relaxation factor for the SOR method (default is 1.75).
    - eta: float. A parameter that controls the influence of the concentration values on the growth probability (default is 1.0).
    - max_sor_iterations: int. The maximum number of iterations for the SOR method (default is 1000).
    - save_snap: bool. Whether to save snapshots of the concentration field at each SOR iteration (default is False).
    
    Returns:
    - cluster: A 2D boolean array representing the final state of the DLA cluster after the specified number of growth steps.
    
    """

    frames = []

    if seed is not None:
        np.random.seed(seed)

    # initialize the concentration field, cluster, and object matrix
    c, cluster, obj_matrix = initialize_dla_laplace(grid_size=grid_size, bottom=bottom, top=top)

    history = {
        "concentration_fields": [],
        "candidate_positions": [],
        "chosen_positions": [],
        "candidate_concentrations": [],
        "growth_probabilities": [],
        "candidate_counts": [],
        "SOR_iterations": []
    }

    # Simulate the growth process for the specified number of steps
    for step in range(steps):
        # Solve the Laplace equation using SOR to get the concentration field
        c_solved, deltas, iteration_used, c_over_time = sor_iteration(
            c, 
            omega=omega,
            max_iteration=max_sor_iterations,
            obj_matrix=obj_matrix,
            epsilon=10**(-5),
            bottom=bottom,
            top=top,
            save_snap=False
        )

        c_solved = np.clip(c_solved, 0, 1)

        # Find candidate positions for growth
        candidates = find_candidates(cluster)
        history["candidate_counts"].append(len(candidates))
        history["SOR_iterations"].append(iteration_used)

        # Choose a position to grow based on the concentration values at the candidate positions
        chosen_pos, probs, cand_list, cand_vals, obj_matrix = growth(
            cluster, 
            obj_matrix, 
            eta, 
            c_solved, 
            candidates
        )

        # Update the history for analysis and visualization
        history["concentration_fields"].append(c_solved.copy())
        history["candidate_positions"].append(cand_list)
        history["chosen_positions"].append(chosen_pos)
        history["candidate_concentrations"].append(cand_vals)
        history["growth_probabilities"].append(probs)

        c = c_solved

        # SAVE FRAME
        if save_every_step:
            frames.append(cluster.astype(np.uint8).copy())

        if progress_every and (step + 1) % progress_every == 0:
            left = steps - (step + 1)
            print(f"Growth step {step+1}/{steps} | cluster size = {cluster.sum()} | SOR iterations = {iteration_used}")

    return cluster, obj_matrix, c, history, frames 


def build_growth_time_grid(chosen_positions, grid_size, interval=25, tail=50):
    """
    This function builds a grid that records the time step at which each position in the cluster was occupied during the growth process.
    The grid is initialized with NaN values to indicate empty positions, and the seed position is set to 0.
    As particles are added to the cluster, the corresponding positions in the grid are updated with the time step at which they were occupied.
    Returns an array where:
    - NaN = empty (white background)
    - 0 = seed (black)
    - 1,2,3,... = time step when particle was added
    """
    T = np.full((grid_size, grid_size), np.nan)

    center = (0, grid_size // 2)
    T[center] = 0  # seed

    for t, pos in enumerate(chosen_positions, start=1):
        T[pos] = t

    max_t = int(np.nanmax(T))
    fig, ax = plt.subplots()

    img = np.full_like(T, np.nan, dtype=float)
    
    im = ax.imshow(
        img, 
        origin="lower", 
        cmap = "rainbow",
        vmin=0, 
        vmax=tail)

    im.cmap.set_bad(color="white")

    def update(k):
        # show sites grown up to time k
        grown = (~np.isnan(T)) & (T <= k)

        # start with white background everywhere
        img = np.full_like(T, np.nan, dtype=float)

        # mark all grown sites as "old"
        img[grown] = -1.0

        # recent sites: age = k - T (0 = newest)
        recent = grown & ((k - T) <= tail)
        img[recent] = (tail - (k - T[recent]))  # newest -> tail, older -> smaller

        # make a masked array so NaNs stay white
        m = np.ma.masked_invalid(img)
        im.set_data(m)

        # black for old points (value -1)
        im.cmap.set_under("black")

        ax.set_title(f"Laplace Growth")
        return (im,)

    ani = FuncAnimation(
        fig, 
        update, 
        frames=max_t + 1, 
        interval=interval, 
        blit=True)

    return ani, fig, update, max_t

def eta_experiment(eta_list, omega, steps, grid_size, seed, progress_every, max_sor_iterations, interval, tail):
    """
    This function runs the DLA simulation for different values of eta and saves the resulting clusters and growth animations.
    """

    for eta in eta_list:
        cluster, obj_matrix, c, hist, frames = diffusion_limited_aggregation(
            steps=steps,
            grid_size=grid_size,
            eta=eta,
            omega=omega,
            seed=seed,
            save_every_step=False,
            progress_every=progress_every,
            max_sor_iterations= max_sor_iterations
        )

        plt.figure(figsize=(6, 6))
        plt.imshow(
            cluster, 
            origin="lower", 
            interpolation = "nearest",
            vmin = 0, 
            vmax = 1,
            cmap="gray_r")
        
        plt.title(f"DLA cluster | eta={eta} | omega={omega}")
        plt.savefig(f"Assignment2/Figures/2.1/dla_cluster_eta_{eta}_omega_{omega}.png", dpi=120)
        plt.close()

        build_growth_time_grid(
                hist["chosen_positions"],
                grid_size=grid_size,
                interval=interval,
                tail=tail,
            )
        
        ani, fig, update, max_t = build_growth_time_grid(
            hist["chosen_positions"],
            grid_size=grid_size,
            interval=interval,
            tail=tail
        )

        gif_name = f"Assignment2/Figures/2.1/eta/dla_growth_eta_{eta}_omega_{omega}.gif"
        ani.save(gif_name, dpi=120, writer="pillow")

        # save final frame png
        update(max_t)
        fig.savefig(f"Assignment2/Figures/2.1/eta/dla_growth_final_eta_{eta}_omega_{omega}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved eta={eta}, omega={omega}")



def omega_experiment(eta, omega_list, steps, grid_size, seed, progress_every, max_sor_iterations, interval, tail):
    """
    This function runs the DLA simulation for different values of omega and saves the resulting clusters and growth animations.
    """
    results = []

    for omega in omega_list:
        cluster, obj_matrix, c, hist, frames = diffusion_limited_aggregation(
            steps=steps,
            grid_size=grid_size,
            eta=eta,
            omega=omega,
            seed=seed,
            save_every_step=False,
            progress_every=progress_every,
            max_sor_iterations= max_sor_iterations
        )

        iters = hist["SOR_iterations"]
        max_iters = max(iters)
        avg_iters = sum(iters) / len(iters)

        results.append((omega, max_iters, avg_iters))
        print(f"Omega={omega} | Max SOR iterations: {max_iters} | Average SOR iterations: {avg_iters:.2f}")

    best_omega = results[0]

    for result in results:
        if result[1] < best_omega[1]:   # compare max iterations
            best_omega = result

    print(f"Best omega value: omega={best_omega[0]} (max={best_omega[1]}, avg={best_omega[2]:.1f})")
    return best_omega, results


def run_single_experiment(eta, omega, steps, grid_size, seed, progress_every, max_sor_iterations, interval, tail):
    """
    This function runs a single DLA simulation for the given parameters and saves the resulting cluster and growth animation.
    """

    cluster, obj_matrix, c, hist, frames = diffusion_limited_aggregation(
        steps=steps,
        grid_size=grid_size,
        eta=eta,
        omega=omega,
        seed=seed,
        save_every_step=False,
        progress_every=progress_every,
        max_sor_iterations= max_sor_iterations
    )

    os.makedirs(f"Assignment2/Figures/2.1/grid/eta_{eta}_omega_{omega}", exist_ok=True)
    os.makedirs(f"Assignment2/Figures/2.1/gif/eta_{eta}_omega_{omega}", exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(
        cluster.astype(int), 
        origin="lower", 
        interpolation = "nearest",
        vmin = 0, 
        vmax = 1,
        cmap="gray_r")
    
    plt.title(f"DLA cluster | eta={eta} | omega={omega}")
    plt.savefig(f"Assignment2/Figures/2.1/grid/eta_{eta}_omega_{omega}/dla_cluster.png", dpi=120)
    plt.savefig(f"Assignment2/Figures/2.1/grid/eta_{eta}_omega_{omega}/dla_cluster.pdf", dpi=120)
    plt.close()


    ani, fig, update, max_t = build_growth_time_grid(
        hist["chosen_positions"],
        grid_size=grid_size,
        interval=interval,
        tail=tail
    )

    ani.save(f"Assignment2/Figures/2.1/gif/eta_{eta}_omega_{omega}/dla_growth.gif", dpi=120, writer="pillow")

    update(max_t)
    fig.savefig(f"Assignment2/Figures/2.1/gif/eta_{eta}_omega_{omega}/dla_growth_final.png",
                dpi=150, bbox_inches="tight")
    
    fig.savefig(f"Assignment2/Figures/2.1/gif/eta_{eta}_omega_{omega}/dla_growth_final.pdf",
                dpi=150, bbox_inches="tight")
    
    plt.close(fig)

    iterations = hist["SOR_iterations"]
    max_iterations = max(iterations)
    average_iterations = sum(iterations) / len(iterations)

    return (eta, omega, max_iterations, average_iterations)    


def parallel_experiment(eta_list, omega_list, steps, grid_size, seed, progress_every, max_sor_iterations, interval, tail, workers):
    """
    This function runs the DLA simulations for all combinations of eta and omega in parallel using multiprocessing.
    """

    combinations = []

    for eta in eta_list:
        for omega in omega_list:
            combinations.append((eta, omega))

    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for eta, omega in combinations:
            futures.append(executor.submit(
                run_single_experiment, 
                eta, 
                omega, 
                steps, 
                grid_size, 
                seed, 
                progress_every, 
                max_sor_iterations, 
                interval, 
                tail
            ))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Completed experiment: eta={result[0]}, omega={result[1]}, max_iters={result[2]}, avg_iters={result[3]:.2f}")
    return results



# #Run the DLA for different eta values and save the results
# eta_list = [0.0, 0.5, 1.0, 1.5, 2.0]
# eta_experiment(
#     eta_list=eta_list,
#     omega=1.9,
#     steps=1000,
#     grid_size=100,
#     seed=0,
#     progress_every=50,
#     max_sor_iterations=1000,
#     interval=50,
#     tail=50
# )


# # Run the DLA for different omega values and save the results
# omega_list = [1.75, 1.8, 1.85, 1.9, 1.95, 2]

# best_omega, results = omega_experiment(
#     eta=1.0,
#     omega_list=omega_list,
#     steps=1000,
#     grid_size=100,
#     seed=0, 
#     progress_every=50,
#     max_sor_iterations=1000,
#     interval=50,
#     tail=50
# )

# omegas = [r[0] for r in results]
# max_iters = [r[1] for r in results]
# avg_iters = [r[2] for r in results]

# plt.figure()
# plt.plot(omegas, avg_iters, marker="o", label="Average iterations")
# plt.plot(omegas, max_iters, marker="o", label="Max iterations")
# plt.xlabel("omega")
# plt.ylabel("SOR iterations")
# plt.title("SOR iterations vs omega")
# plt.legend()
# plt.savefig(f"Assignment2/Figures/2.1/omega/best_omega_{best_omega[0]}.png", dpi=120)
# plt.show()

