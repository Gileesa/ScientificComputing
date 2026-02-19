import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

def sor_iteration(c, omega, max_iteration, obj_matrix, epsilon = 10**(-5), save_snap = False):
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
    c_over_time = []

    for k in range(max_iteration):
        c[0, :] = 0 # set boundary condition
        c[-1, :] = 1 # set boundary condition
        c_old = c.copy()

        for j in range(1, Ny-1):
            for i in range(Nx):
                if obj_matrix[j, i] == 1:
                    c[j, i] = 0
                    continue
                elif obj_matrix[j, i] == 2:
                    continue
                right_ = (i + 1) % Nx # periodic boundary condition 49 + 1 = 50 % 50 = 0
                left_ = (i - 1) % Nx # periodic boundary condition 0 - 1 = -1 % 50 = 49

                right = c[j, right_] if obj_matrix[j, right_] != 2 else c[j, i]
                left = c[j, left_] if obj_matrix[j, left_] != 2 else c[j, i]
                up = c[j + 1, i] if obj_matrix[j + 1, i] != 2 else c[j, i]
                down = c[j - 1, i] if obj_matrix[j - 1, i] != 2 else c[j, i]
                
                neighbour[j, i] = 0.25 * (up + down + right + left)
                c[j, i] = (1 - omega) * c_old[j, i] + omega * neighbour[j, i]
    
        c[0, :] = 0 # set boundary condition
        c[-1, :] = 1 # set boundary condition

        delta = np.max(np.abs(c - c_old)) 
        delta_list.append(delta)

        if save_snap:
            c_over_time.append(c.copy())

        if delta < epsilon:
            print(f"SOR omega={omega}, board size={Ny} converged after {k + 1} iterations.")
            return c, np.array(delta_list), (k + 1), np.array(c_over_time)

    return c, np.array(delta_list), (k + 1), np.array(c_over_time)

def run_sor_diff_vals():
    omega_list = np.linspace(1.7, 2, 11, endpoint=False)[1:]
    N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    k_list = []
    iter_list_50 = []
    best_omegas = []
    for N in N_list:
        c_initial = np.zeros((N, N))
        c_initial[0, :] = 0 # set boundary condition at y = 0 to 0
        c_initial[-1, :] = 1 # set boundary condition at y = 1 to 1
        for omega in omega_list:
            _, _, k, _ = sor_iteration(c_initial.copy(), obj_matrix=np.zeros((N,N)), omega=omega, max_iteration=900)
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
                object_matrix[i, j] = object_type
                object_matrix[i + 1, j] = object_type
                return True, object_matrix
        else: 
            if j < N - 1 and object_matrix[i, j] == 0 and object_matrix[i, j + 1] == 0:
                object_matrix[i, j] = object_type
                object_matrix[i, j + 1] = object_type
                return True, object_matrix
    return False, object_matrix

def create_multiple_obj(object_count, object_type, N):
    object_matrix = np.zeros((N, N))
    placed_obj = 0
    while placed_obj < object_count:
        success, object_matrix = create_objects(object_type, N, object_matrix)
        if success:
            placed_obj += 1
            print(f'placed object {placed_obj} out of {object_count}')
    return object_matrix

def test_diff_omegas(omega_list, obj_count, obj_type, N):
    N = 50
    c_initial = np.zeros((N, N))
    c_initial[0, :] = 0 # set boundary condition at y = 0 to 0
    c_initial[-1, :] = 1 # set boundary condition at y = 1 to 1
    k_list = []
    for omega in omega_list:
        k_list_temp = []
        for i in range(10):
            obj_mat = create_multiple_obj(obj_count, obj_type, N)
            _, _, k, _ = sor_iteration(c_initial.copy(), omega=omega, obj_matrix=obj_mat, max_iteration=900)
            k_list_temp.append(k)
            print(f'finished run {i} out of 10 for obj count {obj_count}')
        k_list.append(np.mean(k_list_temp))
    return k_list

def multiple_runs_obj(obj_count, obj_type, N):
    all_delta_list = []
    k_list = []
    for _ in range(10):
        obj_mat = create_multiple_obj(obj_count, obj_type, N)
        c_sor, delta_list_sor, k, c_over_time = sor_iteration(c_initial.copy(), omega=1.85, obj_matrix=obj_mat, max_iteration=500, save_snap=True)
        delta_list_sor = delta_list_sor.tolist()
        k_list.append(k)
        all_delta_list.append(delta_list_sor)
    avg_k = np.mean(k_list)
    max_len = max(len(d) for d in all_delta_list)
    for i in range(len(all_delta_list)):
        last_val = all_delta_list[i][-1]
        while len(all_delta_list[i]) < max_len:
            all_delta_list[i].append(last_val)
    delta_array = np.array(all_delta_list)
    avg_delta = np.mean(delta_array, axis=0)
    return c_sor, avg_delta, avg_k, obj_mat, c_over_time

def animate_conc(c_over_time, obj_mat, title, name):
    time_step = c_over_time.shape[0]

    fig, ax = plt.subplots()
    
    im = ax.imshow(c_over_time[0], origin="lower", cmap="plasma")
    #ins_mask = np.ma.masked_where(obj_mat == 0, obj_mat)
    #ins_img = ax.imshow(ins_mask, origin="lower", cmap="Reds", alpha=0.5)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Concentration")
    ax.set_title(title)

    def update(t):
        im.set_data(c_over_time[t])
        return im,
    
    ani = FuncAnimation(
        fig, 
        update, 
        frames=time_step, 
        interval=100, 
        blit=True)
    ani.save(f"Figures/{name}.gif", fps=10, dpi=200)
    plt.show()
        

# Try N = 50
N = 50 
c_initial = np.zeros((N, N))
c_initial[0, :] = 0 # set boundary condition at y = 0 to 0
c_initial[-1, :] = 1 # set boundary condition at y = 1 to 1

c_jacobi, delta_list = jacobi_iteration(c_initial.copy(), 5000)
c_gauss, delta_list_gauss = gauss_seidel_iteration(c_initial.copy(), 2500)
c_sor, delta_list_sor, _, _ = sor_iteration(c_initial.copy(), omega=1.85, obj_matrix=np.zeros((N, N)), max_iteration=500)

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
plt.savefig("Figures/1.3/jacobi_gauss_sor_comparison.png")
plt.show()

# Question I 
# Show how the convergence measure δ in eq. (14) 
# depends on the number of iterations k for each of the methods. 
# A log-lin plot may be suitable. 
# For SOR, choose a few representative values for ω.

omegas = [1.7, 1.75, 1.80, 1.85, 1.90, 1.95, 2.0, 2.05]

plt.semilogy(delta_list, label="Jacobi")
plt.semilogy(delta_list_gauss, label="Gauss-Seidel")

# For SOR, choose a few representative values for ω.
for w in omegas:
    _, delta_list_sor = sor_iteration(c_initial.copy(), omega=w, max_iteration=500)
    plt.semilogy(delta_list_sor, label=f"SOR omega={w}")

plt.semilogy(delta_list_sor, label="SOR omega=1.85")
plt.xlabel("Iteration k")
plt.ylabel("Delta")
plt.legend()
plt.savefig("Figures/1.3/convergence_comparison.png")
plt.show()

#Question J
#due to run time making the running of this toggle on and off
#runnging with different N and omega values for plotting seeing the effects
run_j = True
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
    plt.savefig('Figures/1.3/QJ_N_vs_omega.png')
    plt.show()

    plt.plot(omegas, iter_list)
    plt.xlabel("ω")
    plt.ylabel("iteration count")
    plt.grid()
    plt.savefig('Figures/1.3/QJ_omega_vs_k.png')
    plt.show()

#Question K
#number of sink objects vs the iteration
N = 50
#obj_mat1 = create_multiple_obj(1, 1, N)
#obj_mat2 = create_multiple_obj(2, 1, N)
#obj_mat3 = create_multiple_obj(3, 1, N)
#obj_mat4 = create_multiple_obj(4, 1, N)
#obj_mat5 = create_multiple_obj(5, 1, N)
c_sor0, delta_list_sor0, k0, c_over_time0 = sor_iteration(c_initial.copy(), omega=1.85, obj_matrix=np.zeros((N, N)), max_iteration=500, save_snap=True)
#c_sor1, delta_list_sor1, k1, _ = sor_iteration(c_initial.copy(), omega=1.85, obj_matrix=obj_mat1, max_iteration=500)
#c_sor2, delta_list_sor2, k2, _ = sor_iteration(c_initial.copy(), omega=1.85, obj_matrix=obj_mat2, max_iteration=500)
#c_sor3, delta_list_sor3, k3, _ = sor_iteration(c_initial.copy(), omega=1.85, obj_matrix=obj_mat3, max_iteration=500)
#c_sor4, delta_list_sor4, k4, _ = sor_iteration(c_initial.copy(), omega=1.85, obj_matrix=obj_mat4, max_iteration=500)
#c_sor5, delta_list_sor5, k5, _ = sor_iteration(c_initial.copy(), omega=1.85, obj_matrix=obj_mat5, max_iteration=500)
c_sor1, delta_list_sor1, k1, obj_mat1, c_over_time1 = multiple_runs_obj(1, 1, N)
c_sor2, delta_list_sor2, k2, obj_mat2, c_over_time2 = multiple_runs_obj(2, 1, N)
c_sor3, delta_list_sor3, k3, obj_mat3, c_over_time3 = multiple_runs_obj(3, 1, N)
c_sor4, delta_list_sor4, k4, obj_mat4, c_over_time4 = multiple_runs_obj(4, 1, N)
c_sor5, delta_list_sor5, k5, obj_mat5, c_over_time5  = multiple_runs_obj(5, 1, N)
k_list = [k0, k1, k2, k3, k4, k5]
obj_count = [0, 1, 2, 3, 4, 5]
plt.plot(obj_count, k_list)
plt.xlabel("Number of sink objects")
plt.ylabel("Iteration count")
plt.grid()
plt.savefig('Figures/1.3/QK_objcount_vs_k.png')
plt.show()

#convergence rate δ vs the iteration count
plt.plot(delta_list_sor0, color='red', label='0 objects')
plt.plot(delta_list_sor1, color='orange', label='1 object')
plt.plot(delta_list_sor2, color='yellow', label='2 objects')
plt.plot(delta_list_sor3, color='green', label='3 objects')
plt.plot(delta_list_sor4, color='blue', label='4 objects')
plt.plot(delta_list_sor5, color='purple', label='5 objects')
plt.yscale('log')
plt.xlabel("Iteration count")
plt.ylabel("δ")
plt.legend()
plt.grid()
plt.savefig('Figures/1.3/QK_k_vs_delta.png')
plt.show()

run_k_omega = True
#testing out different ω values
if run_k_omega:
    omega_list = np.linspace(1.7, 2, 11, endpoint=False)[1:]
    iter_list0 = test_diff_omegas(omega_list, 0, 0, N)
    iter_list1 = test_diff_omegas(omega_list, 1, 1, N)
    iter_list2 = test_diff_omegas(omega_list, 2, 1, N)
    iter_list3 = test_diff_omegas(omega_list, 3, 1, N)
    iter_list4 = test_diff_omegas(omega_list, 4, 1, N)
    iter_list5 = test_diff_omegas(omega_list, 5, 1, N)
    plt.plot(omega_list, iter_list0, color='red', label='0 objects')
    plt.plot(omega_list, iter_list1, color='orange', label='1 object')
    plt.plot(omega_list, iter_list2, color='yellow', label='2 objects')
    plt.plot(omega_list, iter_list3, color='green', label='3 objects')
    plt.plot(omega_list, iter_list4, color='blue', label='4 objects')
    plt.plot(omega_list, iter_list5, color='purple', label='5 objects')
    plt.xlabel("ω")
    plt.ylabel("iteration count")
    plt.legend()
    plt.grid()
    plt.savefig('Figures/1.3/QK_omega_vs_k.png')
    plt.show()
    best0 = omega_list[iter_list0.index(min(iter_list0))]
    best1 = omega_list[iter_list1.index(min(iter_list1))]
    best2 = omega_list[iter_list2.index(min(iter_list2))]
    best3 = omega_list[iter_list3.index(min(iter_list3))]
    best4 = omega_list[iter_list4.index(min(iter_list4))]
    best5 = omega_list[iter_list5.index(min(iter_list5))]
    print(f'best ω for 0 objects {best0}')
    print(f'best ω for 0 objects {best1}')
    print(f'best ω for 0 objects {best2}')
    print(f'best ω for 0 objects {best3}')
    print(f'best ω for 0 objects {best4}')
    print(f'best ω for 0 objects {best5}')


#showing the concentration plot with the objects
plt.figure()
plt.imshow(c_sor0, origin="lower", cmap="plasma")
plt.colorbar(label="Concentration")
plt.title("Diffusion with 0 objects")
plt.savefig('Figures/1.3/QK_concentration0.png')
plt.show()

animate_conc(c_over_time0, np.zeros((N, N)), 'Diffusion over time with 0 objects', 'no_obj')

plt.figure()
plt.imshow(c_sor1, origin="lower", cmap="plasma")
plt.colorbar(label="Concentration")
obj_mask = np.ma.masked_where(obj_mat1 == 0, obj_mat1)
plt.imshow(obj_mask, origin="lower", cmap="Reds", alpha=0.5)
plt.title("Diffusion with 1 sink object")
plt.savefig('Figures/1.3/QK_concentration1.png')
plt.show()

animate_conc(c_over_time1, obj_mat1, 'Diffusion over time with 1 sink object', 'sink_1_obj')

plt.figure()
plt.imshow(c_sor2, origin="lower", cmap="plasma")
plt.colorbar(label="Concentration")
obj_mask = np.ma.masked_where(obj_mat2 == 0, obj_mat2)
plt.imshow(obj_mask, origin="lower", cmap="Reds", alpha=0.5)
plt.title("Diffusion with 2 sink objects")
plt.savefig('Figures/1.3/QK_concentration2.png')
plt.show()

animate_conc(c_over_time2, obj_mat2, 'Diffusion over time with 2 sink objects', 'sink_2_obj')

plt.figure()
plt.imshow(c_sor3, origin="lower", cmap="plasma")
plt.colorbar(label="Concentration")
obj_mask = np.ma.masked_where(obj_mat3 == 0, obj_mat3)
plt.imshow(obj_mask, origin="lower", cmap="Reds", alpha=0.5)
plt.title("Diffusion with 3 sink objects")
plt.savefig('Figures/1.3/QK_concentration3.png')
plt.show()

animate_conc(c_over_time3, obj_mat3, 'Diffusion over time with 3 sink objects', 'sink_3_obj')

plt.figure()
plt.imshow(c_sor4, origin="lower", cmap="plasma")
plt.colorbar(label="Concentration")
obj_mask = np.ma.masked_where(obj_mat4 == 0, obj_mat4)
plt.imshow(obj_mask, origin="lower", cmap="Reds", alpha=0.5)
plt.title("Diffusion with 4 sink objects")
plt.savefig('Figures/1.3/QK_concentration4.png')
plt.show()

animate_conc(c_over_time4, obj_mat4, 'Diffusion over time with 4 sink objects', 'sink_4_obj')

plt.figure()
plt.imshow(c_sor5, origin="lower", cmap="plasma")
plt.colorbar(label="Concentration")
obj_mask = np.ma.masked_where(obj_mat5 == 0, obj_mat5)
plt.imshow(obj_mask, origin="lower", cmap="Reds", alpha=0.5)
plt.title("Diffusion with 5 sink objects")
plt.savefig('Figures/1.3/QK_concentration5.png')
plt.show()

animate_conc(c_over_time5, obj_mat5, 'Diffusion over time with 5 sink objects', 'sink_5_obj')

#Question L
obj_mat = create_multiple_obj(3, 2, N)
c_sor, _, _, c_over_time = sor_iteration(c_initial.copy(), omega=1.85, obj_matrix=obj_mat, max_iteration=500, save_snap=True)

plt.figure()
plt.imshow(c_sor, origin="lower", cmap="plasma")
plt.colorbar(label="Concentration")
obj_mask = np.ma.masked_where(obj_mat == 0, obj_mat)
plt.imshow(obj_mask, origin="lower", cmap="Reds", alpha=0.5)
plt.title("Diffusion with 3 insulating objects")
plt.savefig('Figures/1.3/QK_concentration_insulating.png')
plt.show()

animate_conc(c_over_time, obj_mat, 'Diffusion over time with 3 insulating objects', 'insulating_3_obj')
