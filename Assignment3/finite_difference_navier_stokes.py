#
# Finite difference simulation of navier stokes
#


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple, List
import matplotlib.animation as animation
import os
import re


def create_grid(nx: int, ny: int, Lx: float, Ly: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Create a uniform rectangular grid.

    Params
    - nx : int
        Number of grid points in x direction
    - ny : int
        Number of grid points in y direction
    - Lx : float
        Domain length in x direction
    - Ly : float
        Domain length in y direction

    Returns:
    - X : np.ndarray
        2D meshgrid x coordinates
    - Y : np.ndarray
        2D meshgrid y coordinates
    - dx : float
        Grid spacing in x direction
    - dy : float
        Grid spacing in y direction
    """

    dx = Lx / (nx - 1) # nx-1 because we have n intervals, n+1 grid points
    dy = Ly / (ny - 1) 

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    X, Y = np.meshgrid(x, y)

    return X, Y, dx, dy


def create_cylinder_mask(X: np.ndarray, Y: np.ndarray,
                         cx: float, cy: float, r: float) -> np.ndarray:
    """
    Create a boolean mask representing a cylindrical obstacle.

    Points inside the cylinder are marked True.

    This allows us to enforce no-slip conditions inside the obstacle.

    Params
    - X, Y : np.ndarray
        Grid coordinates
    - cx, cy : float
        Cylinder center
    - r : float
        Cylinder radius

    Returns:
    - mask : np.ndarray
        Boolean mask for cylinder
    """

    mask = (X - cx)**2 + (Y - cy)**2 <= r**2
    return mask


def pressure_poisson(p: np.ndarray,
                     u: np.ndarray,
                     v: np.ndarray,
                     dx: float,
                     dy: float,
                     dt: float,
                     rho: float,
                     nit: int,
                     cylinder_mask: np.ndarray) -> np.ndarray:
    """
    Solve the pressure Poisson equation.

    This equation arises because incompressible flow requires:

        ∇ · u = 0

    We solve for the pressure field to ensure this.

    Params
    - p : np.ndarray
        Pressure field
    - u : np.ndarray
        x velocity
    - v : np.ndarray
        y velocity
    - dx, dy : float
        Grid spacing
    - dt : float
        Time step
    - rho : float
        Density
    - nit : int
        Number of pressure iterations
    - cylinder_mask : np.ndarray
        Mask for cylinder obstacle

    Returns:
    - p : np.ndarray
        Updated pressure field
    """

    pn = np.empty_like(p)

    for _ in range(nit):

        pn[:] = p[:] # copy pressure array. pn is old, p becomes new

        # Finite difference approximation of Poisson equation
        p[1:-1,1:-1] = (
            ((pn[1:-1,2:] + pn[1:-1,:-2]) * dy**2 +
             (pn[2:,1:-1] + pn[:-2,1:-1]) * dx**2)
            /
            (2 * (dx**2 + dy**2))
            -
            rho * dx**2 * dy**2 /
            (2 * (dx**2 + dy**2))
            *
            (
                (1/dt) *
                ((u[1:-1,2:] - u[1:-1,:-2])/(2*dx) +
                 (v[2:,1:-1] - v[:-2,1:-1])/(2*dy))
                -
                ((u[1:-1,2:] - u[1:-1,:-2])/(2*dx))**2
                -
                2 *
                ((u[2:,1:-1] - u[:-2,1:-1])/(2*dy) *
                 (v[1:-1,2:] - v[1:-1,:-2])/(2*dx))
                -
                ((v[2:,1:-1] - v[:-2,1:-1])/(2*dy))**2
            )
        )

        # Boundary conditions for pressure

        p[:, -1] = p[:, -2]     # dp/dx = 0 at x = Lx (Neumann)
        p[:, 0] = p[:, 1]       # dp/dx = 0 at x = 0 (Neumann)
        p[0, :] = p[1, :]       # dp/dy = 0 at y = 0 (Neumann)
        p[-1, :] = 0            # p = 0 at y = Lx (Dirichlet)

        # enforce constant pressure inside cylinder
        p[cylinder_mask] = 0

    return p


def velocity_update(u: np.ndarray,
                    v: np.ndarray,
                    p: np.ndarray,
                    dx: float,
                    dy: float,
                    dt: float,
                    rho: float,
                    nu: float,
                    cylinder_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update velocity using finite difference Navier-Stokes scheme.

    Includes 5-point finite difference scheme and enforces BCs.

    Params
    - u, v : np.ndarray
        Velocity fields
    - p : np.ndarray
        Pressure field
    - dx, dy : float
        Grid spacing
    - dt : float
        Time step
    - rho : float
        Fluid density
    - nu : float
        Kinematic viscosity
    - cylinder_mask : np.ndarray
        Cylinder mask

    Returns
    - u, v : np.ndarray
        Updated velocity fields
    """

    un = u.copy()
    vn = v.copy()

    # finite difference update
    u[1:-1,1:-1] = (
        un[1:-1,1:-1]
        - un[1:-1,1:-1] * dt/dx * (un[1:-1,1:-1] - un[1:-1,:-2])
        - vn[1:-1,1:-1] * dt/dy * (un[1:-1,1:-1] - un[:-2,1:-1])
        - dt/(rho*2*dx) * (p[1:-1,2:] - p[1:-1,:-2])
        + nu * (
            dt/dx**2 * (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2])
            +
            dt/dy**2 * (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1])
        )
    )

    v[1:-1,1:-1] = (
        vn[1:-1,1:-1]
        - un[1:-1,1:-1] * dt/dx * (vn[1:-1,1:-1] - vn[1:-1,:-2])
        - vn[1:-1,1:-1] * dt/dy * (vn[1:-1,1:-1] - vn[:-2,1:-1])
        - dt/(rho*2*dy) * (p[2:,1:-1] - p[:-2,1:-1])
        + nu * (
            dt/dx**2 * (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2])
            +
            dt/dy**2 * (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1])
        )
    )

    # enforce no-slip condition inside cylinder
    u[cylinder_mask] = 0
    v[cylinder_mask] = 0

    return u, v


def apply_velocity_boundary_conditions(u: np.ndarray,
                                       v: np.ndarray,
                                       U_inlet:float,
                                       cylinder_mask: np.ndarray,) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the boundary conditions in accordance with the assignment.

    Left boundary (inlet): u = U_inlet, v = 0
    Right boundary (outlet): du/dx = 0 
    Top/bottom walls: u=v=0
    Inside cylinder: u=v=0

    Returns:
    - u, v : np.ndarray
        Velocity fields with boundary conditions applied
    """

    # inlet (left)
    u[:,0] = U_inlet
    v[:,0] = 0

    # outlet (right)
    u[:,-1] = u[:,-2]
    v[:,-1] = v[:,-2]

    # top/bottom walls
    u[0,:] = 0
    v[0,:] = 0
    u[-1,:] = 0
    v[-1,:] = 0

    # cylinder interior
    u[cylinder_mask] = 0
    v[cylinder_mask] = 0

    return u, v

def clean_filename(title: str, ext: str = ".png"):
    """
    Turn a title into a safe filename.
    Removes most special characters.
    """
    safe = re.sub(r'[^A-Za-z0-9_\-]', '_', title)
    return safe + ext

def plot_flow(X, Y, u, v, p, Re=None):
    fig = plt.figure(figsize=(14,4), dpi=100)  # wide figure

    # Pressure field
    plt.contourf(X, Y, p, alpha=0.6, cmap=cm.viridis)
    plt.colorbar()

    plt.contour(X, Y, p, colors='k', linewidths=0.5)

    # Quiver
    skip_x = 10
    skip_y = 3

    plt.quiver(X[::skip_y, ::skip_x],
               Y[::skip_y, ::skip_x],
               u[::skip_y, ::skip_x],
               v[::skip_y, ::skip_x],
               scale=5,
               width=0.002)

    # Titles
    title = "Finite Difference Fluid Flow Snapshot"
    if Re is not None:
        title += f"\nRe={Re}"

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)

    plt.gca().set_aspect('equal')

    # Save
    save_folder = "Figures/NS_finite_difference"
    os.makedirs(save_folder, exist_ok=True)

    filename = f"{clean_filename(title)}.png"
    plt.savefig(os.path.join(save_folder, filename))

    plt.tight_layout()
    plt.show()


def animate_flow_heat(X: np.ndarray,
                      Y: np.ndarray,
                      u_hist: list[np.ndarray],
                      v_hist: list[np.ndarray],
                      p_hist: list[np.ndarray],
                      Re: float,
                      interval: int = 50) -> None:
    """
    Animates navier-stokes flow over time in 220x36 grid with off-centered cylinder.

    Params:
    - X : np.ndarray
        2D array of x-coordinates of the computational grid.
    - Y : np.ndarray
        2D array of y-coordinates of the computational grid.
    - u_hist : list of np.ndarray
        List of 2D arrays of the x-component of velocity at each saved timestep.
    - v_hist : list of np.ndarray
        List of 2D arrays of the y-component of velocity at each saved timestep.
    - p_hist : list of np.ndarray
        List of 2D arrays of pressure at each saved timestep (optional for plotting contours).
    - interval : int, optional
        Delay between frames in milliseconds (default is 50).
    """

    fig, ax = plt.subplots(figsize=(11,7), dpi=100)

    # initial velocity magnitude heat map
    speed_init = np.sqrt(u_hist[0]**2 + v_hist[0]**2)
    heat = ax.imshow(speed_init, origin='lower',
                     extent=[X.min(), X.max(), Y.min(), Y.max()],
                     cmap=cm.plasma, alpha=0.8)
    cbar = fig.colorbar(heat, ax=ax)
    cbar.set_label("Velocity magnitude")

    def update(frame: int):
        ax.clear()

        u = u_hist[frame]
        v = v_hist[frame]
        speed = np.sqrt(u**2 + v**2)

        # plot velocity magnitude as heat map
        heat = ax.imshow(speed, origin='lower',
                         extent=[X.min(), X.max(), Y.min(), Y.max()],
                         cmap=cm.plasma, alpha=0.8)

        # cylinder outline
        ax.contour(X, Y, cylinder_mask, colors='black')

        # legend dummy handles
        ax.plot([], [], color='black', linestyle='-', label='Cylinder boundary')
        ax.legend(loc='upper right')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Timestep {frame} (Re={Re})")

        return heat

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(u_hist),
        interval=interval
    )

    # Save last frame
    last_frame = len(u_hist) - 1

    u = u_hist[last_frame]
    v = v_hist[last_frame]
    speed = np.sqrt(u**2 + v**2)

    fig_last, ax_last = plt.subplots(figsize=(11,7), dpi=100)

    heat = ax_last.imshow(speed, origin='lower',
                        extent=[X.min(), X.max(), Y.min(), Y.max()],
                        cmap=cm.plasma, alpha=0.8)

    cbar = fig_last.colorbar(heat, ax=ax_last)
    cbar.set_label("Velocity magnitude")

    # cylinder outline
    ax_last.contour(X, Y, cylinder_mask, colors='black')

    ax_last.set_xlabel("X")
    ax_last.set_ylabel("Y")
    ax_last.set_title(f"Final Timestep {last_frame} (Re={Re})")

    # save
    save_folder = "Figures/NS_finite_difference"
    os.makedirs(save_folder, exist_ok=True)

    filename = f"final_frame_Re_{Re}.png"
    plt.savefig(os.path.join(save_folder, filename))

    plt.close(fig_last)

    plt.show()
    # ani.save("flow_animation_heat.mp4", fps=30)

def run_simulation(u_init: np.ndarray,
                   v_init: np.ndarray,
                   p_init: np.ndarray,
                   dx: float,
                   dy: float,
                   dt: float,
                   rho: float,
                   nu: float,
                   nt: int,
                   cylinder_mask: np.ndarray,
                   D: float,
                   U_inlet: float = 1.0,
                   save_every: int = 1
                  ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Run a 2D finite difference simulation of incompressible flow past a cylinder.

    Params:
    - u_init, v_init, p_init : np.ndarray
        Initial velocity and pressure fields.
    - dx, dy : float
        Grid spacing in x and y.
    - dt : float
        Time step.
    - rho : float
        Fluid density.
    - nu : float
        Kinematic viscosity.
    - nt : int
        Number of time steps.
    - cylinder_mask : np.ndarray
        Boolean mask of cylinder obstacle.
    - U_inlet : float
        Inlet velocity.
    - save_every : int
        How often to save snapshots for animation.

    Returns:
    - u_history, v_history, p_history : lists of np.ndarray
        Snapshots of velocity and pressure fields for animation.
    """

    # Copy initial fields
    u = u_init.copy()
    v = v_init.copy()
    p = p_init.copy()

    # Lists to store snapshots
    u_history: List[np.ndarray] = []
    v_history: List[np.ndarray] = []
    p_history: List[np.ndarray] = []

    for n in range(nt):
        # Solve pressure Poisson equation
        p = pressure_poisson(p, u, v, dx, dy, dt, rho, 50, cylinder_mask)

        # Update velocity fields
        u, v = velocity_update(u, v, p, dx, dy, dt, rho, nu, cylinder_mask)

        # Apply velocity boundary conditions
        u, v = apply_velocity_boundary_conditions(u, v, U_inlet=U_inlet, cylinder_mask=cylinder_mask)

        # Save snapshots for animation
        if n % save_every == 0:
            u_history.append(u.copy())
            v_history.append(v.copy())
            p_history.append(p.copy())

    return u_history, v_history, p_history

# ---------------------------------------------------------
# Main Simulation
# ---------------------------------------------------------

nx, ny = 220, 36
Lx, Ly = 2.20, 0.36

rho = 1

dt = 0.0001
nt = 200

X, Y, dx, dy = create_grid(nx, ny, Lx, Ly)

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

r = 0.05 #radius of cylinder
nu = 0.1

# cylinder obstacle
cylinder_mask = create_cylinder_mask(X, Y, cx=0.155, cy=0.165, r=r) # place cylinder off-centre

# lists for animation
u_history = []
v_history = []
p_history = []

save_every = 1  # store every 1 timesteps


# Run simulation
nu_list = [0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1]
U_inlet_list = [1.0, 1.0, 1.0, 2.0, 7.0, 15.0, 16.0]

for nu, U_in in zip(nu_list, U_inlet_list):
    Re = U_in * (2*r) / nu
    print(f"Simulation with Reynolds number: {Re:.1f}")

    u_history, v_history, p_history = run_simulation(
        u_init=u,
        v_init=v,
        p_init=p,
        dx=dx,
        dy=dy,
        dt=dt,
        rho=rho,
        nu=nu,
        nt=nt,
        cylinder_mask=cylinder_mask,
        D=(2*r),
        U_inlet=U_in,
        save_every=save_every
    )
    # Plot final flow using last snapshot
    plot_flow(X, Y, u_history[-1], v_history[-1], p_history[-1], Re=Re)

    # Animate flow
    animate_flow_heat(X, Y, u_history, v_history, p_history, Re=Re)