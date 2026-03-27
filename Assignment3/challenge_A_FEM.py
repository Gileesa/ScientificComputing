#Assignment_3 

# %%
from ngsolve import *
from ngsolve.webgui import Draw
import ipywidgets as widgets
from netgen.occ import *
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from ngsolve import ds

def make_mesh(maxh=0.07, grading=0.3) -> Mesh:
    """
    Generate a 2D finite element mesh for a channel flow simulation around a circular cylinder.
    Creates a rectangular channel (2.2 × 0.41 m) with a circular obstruction (radius 0.05 m)
    centered at (0.2, 0.15). 

    The mesh is generated using OCCGeometry with specified refinement parameters and 
    includes boundary labels for inlet, outlet, walls, and cylinder surface.
    
    Parameters:
    maxh    : float.Maximum element size for mesh generation. Default is 0.07 m.
    grading : float. Mesh grading factor controlling element size transition. Default is 0.3.
    
    Returns:
    Mesh
        A curved (3rd order) finite element mesh with labeled boundaries:
        - "inlet": Left boundary (inlet)
        - "outlet": Right boundary (outlet)
        - "wall": Top and bottom boundaries
        - "cyl": Circular cylinder surface
    """
    shape = Rectangle(2.2, 0.41).Circle(0.2, 0.15, 0.05).Reverse().Face()
    shape.edges.name = "cyl"
    shape.edges.Min(X).name = "inlet"
    shape.edges.Max(X).name = "outlet"
    shape.edges.Min(Y).name = "wall"
    shape.edges.Max(Y).name = "wall"

    geo  = OCCGeometry(shape, dim=2)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh, grading=grading)).Curve(3)
    print("Boundaries:", mesh.GetBoundaries())
    return mesh

def run_simulation(Re=100, tau=0.001, tend=10.0, maxh=0.07, grading=0.3, folder="ScientificComputing/Assignment3/results_FEM"):
    """
    Solve the incompressible Navier-Stokes equations on the Schäfer-Turek geometry using P3/P2 
    Taylor-Hood finite elements and an IMEX (implicit Stokes + explicit convection) time integrator.
 
    Parameters:
    Re      : Reynolds number  Re = U * D / nu
    tau     : time step (s)
    tend    : end time (s)
    maxh    : maximum mesh element size
    grading : mesh grading near curved boundaries
    folder  : directory for saved figures / animations
    """
    os.makedirs(folder, exist_ok=True)
 
    # parameters
    D  = 0.1                   # cylinder diameter [m]
    U_mean  = 1.0                   # mean inflow velocity (used in Re definition)
    nu = U_mean * D / Re            # kinematic viscosity  [m²/s]
    U_max = 1.5
    print(f"\n Re = {Re}  nu = {nu:.5f}")
 
    # Mesh 
    mesh = make_mesh(maxh=maxh, grading=grading)
 
    # FE spaces: P3 velocity, P2 pressure (Taylor-Hood)
    V = VectorH1(mesh, order=3, dirichlet="wall|cyl|inlet")
    Q = H1(mesh, order=2)
    X = V * Q
 
    u, p = X.TrialFunction()
    v, q = X.TestFunction()
 
    # Stokes bilinear form 
    stokes = (nu * InnerProduct(grad(u), grad(v))
              + div(u) * q + div(v) * p
              - 1e-10 * p * q) * dx        # small penalty stabilises pressure
 
    a = BilinearForm(stokes).Assemble()
    f = LinearForm(X).Assemble()           # zero body force
 
    gfu = GridFunction(X)
 
    # Parabolic inflow  u_max = 1.5 U mean = 1.0 U  (Poiseuille)
    # u_max(y) = 1.5 * 4y(H-y)/H²   with H = 0.41

    uin = CoefficientFunction((1.5 * 4 * y * (0.41 - y) / (0.41**2), 0))
    gfu.components[0].Set(uin, definedon=mesh.Boundaries("inlet"))
 
    # Solve Stokes for initial condition 
    inv_stokes = a.mat.Inverse(X.FreeDofs())
    res = f.vec - a.mat * gfu.vec
    gfu.vec.data += inv_stokes * res
 
    # IMEX mass matrix:  M* = M + tau * A  (implicit Stokes)
    mstar = BilinearForm(u * v * dx + tau * stokes).Assemble()
    inv   = mstar.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")
 
    # Non-linear convection term  ∫ (u·∇u) · v dx 
    conv = BilinearForm(X, nonassemble=True)
    conv += (Grad(u) * u) * v * dx         # FIX: was missing entirely
 
    # Storage for drag/lift history and animation snapshots
    times      = []
    drag_hist  = []
    lift_hist  = []
    vel        = gfu.components[0]
    gfut       = GridFunction(V, multidim=0)   # animation snapshots
 
    # Time loop
    t = 0.0
    i = 0
    diverged = False
 
    tw = widgets.Text(value=f"Re={Re}  t=0.00")
    display(tw)
 
    with TaskManager():
        while t < tend:
            # IMEX step: explicit convection + implicit Stokes
            res = conv.Apply(gfu.vec) + a.mat * gfu.vec
            gfu.vec.data -= tau * inv * res
 
            t += tau
            i += 1
 
            # Stability check: if velocity blows up, stop early
            umax = sqrt(InnerProduct(vel.vec, vel.vec))
            if umax > 1e6:
                print(f"  DIVERGED at t={t:.4f}  (|u|={umax:.2e})")
                diverged = True
                break
 
            # Drag and lift on the cylinder every 10 steps 
            if i % 10 == 0:
                # Cauchy stress tensor:  σ = ν(∇u + ∇uᵀ) − pI
                stress = nu * (grad(vel) + grad(vel).trans) \
                         - gfu.components[1] * Id(2)
                n      = specialcf.normal(2)
                
                F_drag = Integrate(stress * n, mesh, BND, definedon=mesh.Boundaries("cyl"))

                drag = - F_drag[0]
                lift = - F_drag[1]
                C_D  = 2 * drag / (U_mean**2 * D)
                C_L  = 2 * lift / (U_mean**2 * D)
                times.append(t)
                drag_hist.append(C_D)
                lift_hist.append(C_L)
 
            # Save snapshot every 50 steps for animation
            if i % 50 == 0:
                gfut.AddMultiDimComponent(vel.vec)
 
            # Update progress label every 100 steps
            if i % 100 == 0:
                tw.value = f"Re={Re}  t={t:.3f}  C_D={drag_hist[-1]:.3f}"
 
    tw.value = f"Re={Re}  {'DIVERGED' if diverged else 'done'}  t={t:.3f}"
 
    # Post-processing
    _plot_forces(times, drag_hist, lift_hist, Re, folder)
    _save_final_velocity(mesh, vel, Re, t, folder)
    _create_animation(mesh, gfut, Re, tau, folder)
 
    return {
        "Re": Re,
        "diverged": diverged,
        "t_final": t,
        "C_D_mean": float(np.mean(drag_hist[-100:])) if drag_hist else None,
        "C_L_max":  float(np.max(np.abs(lift_hist[-100:]))) if lift_hist else None,
    }
 
 
# 3.  Post-processing helpers
def _plot_forces(times, drag_hist, lift_hist, Re, folder):
    """Plot drag and lift coefficient vs. time."""
    if not times:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax1.plot(times, drag_hist, color="steelblue")
    ax1.set_ylabel("C_D")
    ax1.set_title(f"Drag and lift — Re={Re}")
    ax2.plot(times, lift_hist, color="tomato")
    ax2.set_ylabel("C_L")
    ax2.set_xlabel("t [s]")
    plt.tight_layout()
    plt.savefig(f"{folder}/forces_Re{Re}.png", dpi=150)
    plt.savefig(f"{folder}/forces_Re{Re}.pdf")
    plt.close()
    print(f"  Saved force history → {folder}/forces_Re{Re}.png")
 
 
def _save_final_velocity(mesh, vel, Re, tend, folder, nx=220):
    """Save a contour plot of the final velocity magnitude."""
    x_vals = np.linspace(0, 2.2, nx)
    y_vals = np.linspace(0, 0.41, int(nx * 0.41 / 2.2))
    Xg, Yg = np.meshgrid(x_vals, y_vals)
    speed  = np.zeros_like(Xg)
 
    for iy, yv in enumerate(y_vals):
        for ix, xv in enumerate(x_vals):
            try:
                v = vel(mesh(xv, yv))
                speed[iy, ix] = np.sqrt(v[0]**2 + v[1]**2)
            except Exception:
                speed[iy, ix] = 0.0
 
    angle = np.linspace(0, 2 * np.pi, 100)
    fig, ax = plt.subplots(figsize=(12, 3), dpi=150)
    cf = ax.contourf(Xg, Yg, speed, levels=50, cmap="viridis")
    plt.colorbar(cf, ax=ax, label="|u| [m/s]")
    # FIX: cylinder is at (0.2, 0.15) not (0.2, 0.2)
    ax.plot(0.2 + 0.05 * np.cos(angle),
            0.15 + 0.05 * np.sin(angle), "w-", lw=1.5)
    ax.set_title(f"Kármán vortex street — Re={Re}, t={tend:.2f}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    plt.tight_layout()
    plt.savefig(f"{folder}/velocity_Re{Re}.png", dpi=150)
    plt.savefig(f"{folder}/velocity_Re{Re}.pdf")
    plt.close()
    print(f"  Saved velocity field  → {folder}/velocity_Re{Re}.png")
 
 
def _create_animation(mesh, gfut, Re, tau, folder, nx=150):
    """Save an animated GIF of stored velocity snapshots."""
    n_frames = len(gfut.vecs)
    if n_frames == 0:
        print("No animation frames stored.")
        return
 
    x_vals = np.linspace(0, 2.2, nx)
    y_vals = np.linspace(0, 0.41, int(nx * 0.41 / 2.2))
    Xg, Yg = np.meshgrid(x_vals, y_vals)
    angle  = np.linspace(0, 2 * np.pi, 100)
 
    # Pre-compute all frames
    frames = []
    for k in range(n_frames):
        gfut.vec.FV().NumPy()[:] = gfut.vecs[k].FV().NumPy()
        speed = np.zeros_like(Xg)
        for iy, yv in enumerate(y_vals):
            for ix, xv in enumerate(x_vals):
                try:
                    val = gfut(mesh(xv, yv))
                    speed[iy, ix] = np.sqrt(val[0]**2 + val[1]**2)
                except Exception:
                    speed[iy, ix] = 0.0
        frames.append(speed)
    
    print(f"Frame 0 max speed: {frames[0].max():.4f}")
    print(f"Last frame max speed: {frames[-1].max():.4f}")

    vmax = max(f.max() for f in frames)
 
    fig, ax = plt.subplots(figsize=(12, 3), dpi=100)
 
    def update(k):
        ax.clear()
        ax.contourf(Xg, Yg, frames[k], levels=50, cmap="plasma", vmin=0, vmax=vmax)
        ax.plot(0.2 + 0.05 * np.cos(angle),
                0.15 + 0.05 * np.sin(angle), "w-", lw=1.5)   
        ax.set_title(f"Kármán vortex — Re={Re}, t={k * 50 * tau:.2f}")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
 
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
    gif  = f"{folder}/karman_Re{Re}.gif"
    anim.save(gif, fps=20, writer="pillow")
    plt.close()
    print(f"  Saved animation {gif}")
 
 
