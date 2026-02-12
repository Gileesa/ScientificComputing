import math
import numpy as np

# Question H 
# Implement the Jacobi iteration, the Gauss-Seidel method and SOR.
# Test the methods by comparing the result to the analytical result in eq. (5), i.e. the linear dependence of the concentration on y.

# Implement the Jacobi iteration
def jacobi_iteration(c, max_iteration, epsilon = 10**(-5)):
    """
    This function performs the Jacobi iteration for solving the 2D Laplace equation with given boundary conditions.
    params:
    c: the initial matrix with boundary conditions set
    N: the number of x and y steps
    epsioln: the convergence criterion for the iteration
    returns:
    c: the matrix after convergence    

    array[y, x] -> array[row, column] -> array[j, i]
    """
    c_old = c.copy()
    c_new = c.copy()
    Ny, Nx = c.shape

    c_old[0, :] = 0 # set boundary condition
    c_old[-1, :] = 1 # set boundary condition

    delta_list = []

    for k in range(max_iteration):

        for j in range(1, Ny-1):
            for i in range(Nx):
                right = (i + 1) % Nx # periodic boundary condition 49 + 1 = 50 % 50 = 0
                left = (i - 1) % Nx # periodic boundary condition 0 - 1 = -1 % 50 = 49
            
                c_new[j, i] = 0.25 * (c_old[j + 1, i] + c_old[j - 1, i] + c_old[j, right] + c_old[j, left])
        
                # if k == 49 and j == 2 and i == 3:
        print(f"\nIteration {k}")
        print(c_new)

        c_new[0, :] = 0 # set boundary condition
        c_new[-1, :] = 1 # set boundary condition

        delta = np.max(np.abs(c_new - c_old)) 
        delta_list.append(delta)

        if delta < epsilon:
            print(f"Converged after {k} iterations.")
            return c_new, np.array(delta_list)
        
        c_old, c_new = c_new, c_old

    return c_new, np.array(delta_list)


# Implement the Gauss-Seidel iteration
def gauss_seidel_iteration(c, max_iteration, epsilon = 10**(-5)):
    """
    This function performs the Gauss-Seidel iteration for solving the 2D Laplace equation with given boundary conditions.
    params:
    c: the initial matrix with boundary conditions set
    N: the number of x and y steps
    epsioln: the convergence criterion for the iteration
    returns:
    c: the matrix after convergence    

    array[y, x] -> array[row, column] -> array[j, i]
    """
    Ny, Nx = c.shape

    c_old[0, :] = 0 # set boundary condition
    c_old[-1, :] = 1 # set boundary condition

    delta_list = []

    for k in range(max_iteration):
        c_old = c.copy()

        for j in range(1, Ny-1):
            for i in range(Nx):
                right = (i + 1) % Nx # periodic boundary condition 49 + 1 = 50 % 50 = 0
                left = (i - 1) % Nx # periodic boundary condition 0 - 1 = -1 % 50 = 49
            
                c[j, i] = 0.25 * (c[j + 1, i] + c[j - 1, i] + c[j, right] + c[j, left])
        
        print(f"\nIteration {k}")
        print(c)

        c[0, :] = 0 # set boundary condition
        c[-1, :] = 1 # set boundary condition

        delta = np.max(np.abs(c - c_old)) 
        delta_list.append(delta)

        if delta < epsilon:
            print(f"Converged after {k} iterations.")
            return c, np.array(delta_list)

    return c, np.array(delta_list)

# Try N = 50
N = 50 
c_initial = np.zeros((N + 1, N + 1))
c_jacobi, delta_list = jacobi_iteration(c_initial, 5000)
c_gauss, delta_list_gauss = gauss_seidel_iteration(c_initial, 5000)

  
      
    