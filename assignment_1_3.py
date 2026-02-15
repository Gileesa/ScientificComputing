import math
import numpy as np
import matplotlib.pyplot as plt
import random

# Question H 
# Implement the Jacobi iteration, the Gauss-Seidel method and SOR.

# Implement the Jacobi iteration
def jacobi_iteration(c, max_iteration, epsilon = 10**(-5)):
    """
    This function performs the Jacobi iteration for solving the 2D Laplace equation with given boundary conditions.

    params:
    - c: the initial matrix with boundary conditions set
    - N: the number of x and y steps
    - epsioln: the convergence criterion for the iteration

    returns:
    - c: the matrix after convergence    

    array[y, x] -> array[row, column] -> array[j, i]
    """
    c_old = c.copy()
    c_new = c.copy()
    Ny, Nx = c.shape

    c_old[0, :] = 0 # set boundary condition
    c_old[-1, :] = 1 # set boundary condition

    delta_list = []

    for k in range(max_iteration):
        c_old[0, :] = 0 # set boundary condition
        c_old[-1, :] = 1 # set boundary condition

        for j in range(1, Ny-1):
            for i in range(Nx):
                right = (i + 1) % Nx # periodic boundary condition 49 + 1 = 50 % 50 = 0
                left = (i - 1) % Nx # periodic boundary condition 0 - 1 = -1 % 50 = 49
            
                c_new[j, i] = 0.25 * (c_old[j + 1, i] + c_old[j - 1, i] + c_old[j, right] + c_old[j, left])
        
                # if k == 49 and j == 2 and i == 3:
        # print(f"\nIteration {k}")
        # print(c_new)

        c_new[0, :] = 0 # set boundary condition
        c_new[-1, :] = 1 # set boundary condition

        delta = np.max(np.abs(c_new - c_old)) 
        delta_list.append(delta)

        if delta < epsilon:
            print(f"Jacobi Converged after {k} iterations.")
            return c_new, np.array(delta_list)
        
        c_old, c_new = c_new, c_old

    return c_new, np.array(delta_list)


# Implement the Gauss-Seidel iteration
def gauss_seidel_iteration(c, max_iteration, epsilon = 10**(-5)):
    """
    This function performs the Gauss-Seidel iteration for solving the 2D Laplace equation with given boundary conditions.

    params:
    - c: the initial matrix with boundary conditions set
    - N: the number of x and y steps
    - epsilon: the convergence criterion for the iteration
    
    returns:
    - c: the matrix after convergence    

    array[y, x] -> array[row, column] -> array[j, i]
    """
    Ny, Nx = c.shape

    c[0, :] = 0 # set boundary condition
    c[-1, :] = 1 # set boundary condition

    delta_list = []

    for k in range(max_iteration):
        c[0, :] = 0 # set boundary condition
        c[-1, :] = 1 # set boundary condition
        c_old = c.copy()

        for j in range(1, Ny-1):
            for i in range(Nx):
                right = (i + 1) % Nx # periodic boundary condition 49 + 1 = 50 % 50 = 0
                left = (i - 1) % Nx # periodic boundary condition 0 - 1 = -1 % 50 = 49
            
                c[j, i] = 0.25 * (c[j + 1, i] + c[j - 1, i] + c[j, right] + c[j, left])
        
        # print(f"\nIteration {k}")
        # print(c)

        c[0, :] = 0 # set boundary condition
        c[-1, :] = 1 # set boundary condition

        delta = np.max(np.abs(c - c_old)) 
        delta_list.append(delta)

        if delta < epsilon:
            print(f"Gauss Seidel Converged after {k + 1} iterations.")
            return c, np.array(delta_list)

    return c, np.array(delta_list)


# Implement the SOR iteration

def sor_iteration(c, omega, max_iteration, obj_matrix, epsilon = 10**(-5)):
    """
    This function performs the SOR iteration for solving the 2D Laplace equation with given boundary conditions.

    params:
    - c: the initial matrix with boundary conditions set
    - N: the number of x and y steps
    - epsilon: the convergence criterion for the iteration
    returns:
    - c: the matrix after convergence    

    array[y, x] -> array[row, column] -> array[j, i]

    flow of the iteration:
    - c = entire matrix at iteration k
    - c_old = entire matrix at iteration k-1
    - neighbour = the average of the 4 neighbours of c[j, i]
    - c = entire matrix at iteration k + 1
    """
    Ny, Nx = c.shape
    
    c[0, :] = 0 # set boundary condition
    c[-1, :] = 1 # set boundary condition
    neighbour = np.zeros_like(c)
    delta_list = []

    for k in range(max_iteration):
        c[0, :] = 0 # set boundary condition
        c[-1, :] = 1 # set boundary condition
        c_old = c.copy()

        for j in range(1, Ny-1):
            for i in range(Nx):
                if obj_matrix[j, i] == 1:
                    c[j, i] == 0
                    continue
                elif obj_matrix[j, i] == 2:
                    continue
                right = (i + 1) % Nx # periodic boundary condition 49 + 1 = 50 % 50 = 0
                left = (i - 1) % Nx # periodic boundary condition 0 - 1 = -1 % 50 = 49
            
                neighbour[j, i] = 0.25 * (c[j + 1, i] + c[j - 1, i] + c[j, right] + c[j, left])
                c[j, i] = (1 - omega) * c_old[j, i] + omega * neighbour[j, i]
    
        c[0, :] = 0 # set boundary condition
        c[-1, :] = 1 # set boundary condition

        delta = np.max(np.abs(c - c_old)) 
        delta_list.append(delta)

        if delta < epsilon:
            print(f"SOR omega={omega}, board size={Ny} converged after {k + 1} iterations.")
            return c, np.array(delta_list), (k + 1)

    return c, np.array(delta_list), (k + 1)

def run_sor_diff_vals():
    omega_list = np.linspace(1.7, 2, 11, endpoint=False)[1:]
    N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    k_list = []
    iter_list_50 = []
    best_omegas = []
    for N in N_list:
        c_initial = np.zeros((N + 1, N + 1))
        c_initial[0, :] = 0 # set boundary condition at y = 0 to 0
        c_initial[-1, :] = 1 # set boundary condition at y = 1 to 1
        for omega in omega_list:
            _, _, k = sor_iteration(c_initial.copy(), omega=omega, max_iteration=900)
            k_list.append(k)
            if N == 50:
                iter_list_50.append(k)
        max_val = min(k_list)
        best_omegas.append(omega_list[k_list.index(max_val)])
        k_list = []
    return best_omegas, N_list, iter_list_50, omega_list

def create_objects(object_type, N, object_matrix, max_attempts = 100):
    for _ in range(max_attempts):
        orientation = np.random.choice([0, 1])
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if orientation == 0:
            if i < N - 1 and object_matrix[i, j] == 0 and object_matrix[i + 1, j] == 0:
                object_matrix[i, j] == object_type
                object_matrix[i + 1, j] == object_type
                return True, object_matrix
        else: 
            if j < N - 1 and object_matrix[i, j] == 0 and object_matrix[i, j + 1] == 0:
                object_matrix[i, j] == object_type
                object_matrix[i, j + 1] == object_type
                return True, object_matrix
    return False, object_matrix

def create_multiple_obj(object_count, object_type, N):
    object_matrix = np.zeros((N, N))
    placed_obj = 0
    while placed_obj < object_count:
        success, object_matrix = create_objects(object_type, N, object_matrix)
        if success:
            placed_obj += 1
    return object_matrix


# Try N = 50
N = 50 
c_initial = np.zeros((N, N))
c_initial[0, :] = 0 # set boundary condition at y = 0 to 0
c_initial[-1, :] = 1 # set boundary condition at y = 1 to 1

c_jacobi, delta_list = jacobi_iteration(c_initial.copy(), 5000)
c_gauss, delta_list_gauss = gauss_seidel_iteration(c_initial.copy(), 2500)
c_sor, delta_list_sor, _ = sor_iteration(c_initial.copy(), omega=1.85, obj_matrix=np.zeros((N, N)), max_iteration=500)

# H Test the methods by comparing the result to the analytical result in eq. (5), i.e. the linear dependence of the concentration on y.
Ny, Nx = c_jacobi.shape
y = np.linspace(0, 1, Ny)
c_analytical = y
c_numerical_jacobi = c_jacobi
error = np.max(np.abs(c_numerical_jacobi.mean(axis=1) - c_analytical))
print(f"Max error for Jacobi method: {error}")


# H Test the methods by comparing the result to the analytical result in eq. (5), i.e. the linear dependence of the concentration on y.
Ny, Nx = c_gauss.shape
y = np.linspace(0, 1, Ny)
c_analytical = y
c_numerical_gauss = c_gauss
error = np.max(np.abs(c_numerical_gauss.mean(axis=1) - c_analytical))
print(f"Max error for Gauss-Seidel method: {error}")


# H Test the methods by comparing the result to the analytical result in eq. (5), i.e. the linear dependence of the concentration on y.
Ny, Nx = c_sor.shape
y = np.linspace(0, 1, Ny)
c_analytical = y
c_numerical_sor = c_sor
error = np.max(np.abs(c_numerical_sor.mean(axis=1) - c_analytical))
print(f"Max error for SOR method: {error}")

# Verification of the results by plotting the numerical solutions and the analytical solution on the same graph
y = np.linspace(0, 1, Ny)
plt.plot(y, c_jacobi.mean(axis=1), label="Jacobi")
plt.plot(y, c_gauss.mean(axis=1), label="Gauss-Seidel")
plt.plot(y, c_sor.mean(axis=1), label="SOR")
plt.plot(y, y, "--", label="Analytical c=y")
plt.legend()
plt.xlabel("y")
plt.ylabel("c")
plt.savefig("jacobi_gauss_sor_comparison.png")
plt.show()

# Question I 
# Show how the convergence measure δ in eq. (14) 
# depends on the number of iterations k for each of the methods. 
# A log-lin plot may be suitable. 
# For SOR, choose a few representative values for ω.

plt.semilogy(delta_list, label="Jacobi")
plt.semilogy(delta_list_gauss, label="Gauss-Seidel")
plt.semilogy(delta_list_sor, label="SOR omega=1.85")
plt.xlabel("Iteration k")
plt.ylabel("Delta")
plt.legend()
plt.savefig("convergence_comparison.png")
plt.show()

#Question J
#due to run time making the running of this toggle on and off
#runnging with different N and omega values for plotting seeing the effects
run_j = False
if run_j:
    best_omegas, N_vals, iter_list, omegas = run_sor_diff_vals()
    print(iter_list)
    print(best_omegas)
    print(N_vals)
    print(omegas)
    plt.plot(N_vals, best_omegas)
    plt.xlabel("Different board sizes")
    plt.ylabel("ω")
    plt.grid()
    plt.show()

    plt.plot(omegas, iter_list)
    plt.xlabel("ω")
    plt.ylabel("iteration count")
    plt.grid()
    plt.show()

#Question K
N = 50
obj_mat = create_multiple_obj(1, N, 1)
c_sor1, delta_list_sor1, _ = sor_iteration(c_initial.copy(), omega=1.85, obj_matrix=obj_mat, max_iteration=500)
obj_mat = create_multiple_obj(2, N, 1)
c_sor2, delta_list_sor2, _ = sor_iteration(c_initial.copy(), omega=1.85, obj_matrix=obj_mat, max_iteration=500)
obj_mat = create_multiple_obj(3, N, 1)
c_sor3, delta_list_sor3, _ = sor_iteration(c_initial.copy(), omega=1.85, obj_matrix=obj_mat, max_iteration=500)
