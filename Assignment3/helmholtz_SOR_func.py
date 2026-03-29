

# DO NOT RUN THIS FILE
# THIS IS ONLY TO STORE THIS FUNCTION I DID NOT WANT TO DELETE IT BUT WE NEED TO GET IT OUT OF THE WAY

#main loop helmholtz run
def helmholtz_SOR(k, fxy, nx, ny, dx, wall_mask, max_run = 50000):
    u = np.zeros((nx, ny), dtype=np.complex128)
    u_neighbourhood = np.zeros((nx, ny), dtype=np.complex128)
    omega = 0.6
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
            + fxy[1:-1, 1:-1] * dx**2) 
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