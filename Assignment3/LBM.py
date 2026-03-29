"""
Lattice Boltzmann Method (LBM) — Karman Vortex Street Solver
=============================================================
A minimal, educational 2D LBM solver using the D2Q9 lattice and BGK collision
operator.  This code is prepared for the student of the course Scientific Computing at UvA.
The code simulates flow past a circular cylinder at Reynolds number 150 to produce the
classic Karman vortex street.

Note: Not the exact implementation of the benchmark case from the original paper, 
but a simplified version that captures the essential physics and flow features.  
The code is structured for clarity and educational purposes, not for maximum 
performance or accuracy.

Algorithm overview (each timestep):
  1. Compute macroscopic quantities (density, velocity) from distributions
  2. Collision step  — relax f toward local equilibrium (BGK)
  3. Bounce-back    — reflect populations at obstacle nodes
  4. Streaming step — propagate f_i along lattice velocity c_i
  5. Boundary conditions — Zou-He inlet, open outlet

Dependencies: numpy, matplotlib
Usage:        python lbm_karman.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from numba import njit, prange

def run_LBM():
    # =============================================================================
    # 1.  D2Q9 Lattice Definition
    # =============================================================================
    #
    #   6  2  5        Lattice velocities c_i (i = 0..8):
    #    \ | /           0: rest          (0, 0)
    #   3--0--1          1-4: axis-aligned  (±1,0), (0,±1)
    #    / | \           5-8: diagonals     (±1,±1)
    #   7  4  8
    #
    #  Each row of `c` is a velocity vector [cx, cy] for direction i.

    c = np.array([[0, 0],    # 0  — rest
                [1, 0],    # 1  — east
                [0, 1],    # 2  — north
                [-1, 0],   # 3  — west
                [0, -1],   # 4  — south
                [1, 1],    # 5  — north-east
                [-1, 1],   # 6  — north-west
                [-1, -1],  # 7  — south-west
                [1, -1]])  # 8  — south-east

    # Lattice weights (from the D2Q9 equilibrium derivation)
    w = np.array([4/9,                        # rest
                1/9, 1/9, 1/9, 1/9,         # axis-aligned
                1/36, 1/36, 1/36, 1/36])    # diagonals

    # Opposite direction index for each i (used in bounce-back)
    # e.g. opposite of 1 (east) is 3 (west)
    opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    # =============================================================================
    # 2.  Simulation Parameters
    # =============================================================================
    def choose_params(Re_target, Ma_max=0.1, factor = 1):
        Nx = int(880 * factor)
        Ny = int(164 * factor)
        r_cyl = int(20 * factor)
        cx_cyl = int(79 * factor)
        cy_cyl = int(79 * factor)
        cs = 1/np.sqrt(3)
        D = 2 * r_cyl
        tau = 0.55
        U_req = Re_target * (tau - 0.5) / (3 * D)
        U_cap = Ma_max * cs
        U = min(U_req, U_cap)
        Re_ach = U * D / ((tau - 0.5)/3)
        return U, Re_ach, Nx, Ny, r_cyl, cx_cyl, cy_cyl

    def get_params(Re_targets, tau_nom=0.6, Ma_max=0.1):
        U_list = []
        Re_list = []
        Nx_list = []
        Ny_list = []
        cx_cyl_list = []
        cy_cyl_list = []
        r_cyl_list = []  
        Re_ach = 0
        factor = 1
        for Re in Re_targets:
            U, Re_ach, Nx, Ny, r_cyl, cx_cyl, cy_cyl = choose_params(Re, factor = factor)
            while Re_ach < Re:
                factor += 0.25
                U, Re_ach, Nx, Ny, r_cyl, cx_cyl, cy_cyl = choose_params(Re, factor = factor)
            U_list.append(U)
            Re_list.append(Re_ach)
            Nx_list.append(Nx)
            Ny_list.append(Ny)
            cx_cyl_list.append(cx_cyl)
            cy_cyl_list.append(cy_cyl)
            r_cyl_list.append(r_cyl) 
        return U_list, Re_list, Nx_list, Ny_list, r_cyl_list, cx_cyl_list, cy_cyl_list



    #Nx = 880 # domain length (lattice units)
    #Ny = 164 # domain height (lattice units)
    # Cylinder geometry
    #cx_cyl = 79 # cylinder center x (1/5 from inlet)
    #cy_cyl = 79 # cylinder center y (centered vertically)
    #r_cyl = 20 # cylinder radius
    #D = 2 * r_cyl

    # Flow parameters
    #U_inlet = 0.08      # inlet velocity (lattice units, keep ≪ 1 for low Mach)
    Re_targets = [50, 100, 150, 400, 500]       # target Reynolds number

    # Derived quantities:
    #   Re = U * D / nu   →  nu = U * D / Re
    #   In LBM:  nu = cs² * (tau - 0.5)  where cs² = 1/3
    #   Therefore:  tau = 3 * nu + 0.5
                                # cylinder diameter
                        # kinematic viscosity
    tau = 0.55                        # BGK relaxation time
    C_smag = 0.15                     # Smagorinsky constant for subgrid-scale model

    # =============================================================================
    # 3.  Equilibrium Distribution Function
    # =============================================================================

    @njit(cache=True, fastmath=True)
    def equilibrium(rho, ux, uy, Nx, Ny):
        """
        Compute the equilibrium distribution f^eq for the D2Q9 lattice.

        The equilibrium is derived from a second-order Taylor expansion of the
        Maxwell-Boltzmann distribution:

            f_i^eq = w_i * rho * (1 + c_i·u/cs² + (c_i·u)²/(2·cs⁴) - u·u/(2·cs²))

        where cs² = 1/3  (lattice speed of sound squared).

        Parameters
        ----------
        rho : ndarray (Nx, Ny)   — macroscopic density
        ux  : ndarray (Nx, Ny)   — x-component of velocity
        uy  : ndarray (Nx, Ny)   — y-component of velocity

        Returns
        -------
        feq : ndarray (Nx, Ny, 9) — equilibrium distributions
        """
        feq = np.zeros((Nx, Ny, 9), dtype=np.float64)
        usqr = ux**2 + uy**2                 # |u|²

        for i in range(9):
            cu = c[i, 0] * ux + c[i, 1] * uy  # c_i · u
            feq[:, :, i] = w[i] * rho * (1.0
                                        + 3.0 * cu            # c_i·u / cs²
                                        + 4.5 * cu**2         # (c_i·u)² / (2·cs⁴)
                                        - 1.5 * usqr)         # -|u|² / (2·cs²)
        return feq


    @njit(parallel=True, cache=True)
    def stream(f_out, f, c):
        """Numba-accelerated parallel streaming (replaces 9 np.roll calls)."""
        Nx, Ny = f.shape[0], f.shape[1]
        for i in range(9):
            cx_i = c[i, 0]
            cy_i = c[i, 1]
            for x in prange(Nx):
                for y in range(Ny):
                    xs = x - cx_i
                    ys = y - cy_i
                    if xs < 0: xs += Nx
                    elif xs >= Nx: xs -= Nx
                    if ys < 0: ys += Ny
                    elif ys >= Ny: ys -= Ny
                    f[x, y, i] = f_out[xs, ys, i]


    # =============================================================================
    # 5.  Main Function
    # =============================================================================

    def find_obstacle(Nx, Ny, r_cyl, cx_cyl, cy_cyl):
        # =============================================================================
        # 4.  Obstacle Mask  (circular cylinder)
        # =============================================================================
        # Boolean array: True where the obstacle is located
        x = np.arange(Nx)
        y = np.arange(Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')    # X,Y have shape (Nx, Ny)
        obstacle = (X - cx_cyl)**2 + (Y - cy_cyl)**2 <= r_cyl**2
        return obstacle

    def main():
        """Run the LBM simulation: initialization, time loop, and visualization."""

        # =================================================================
        # 6.  Visualization Setup
        # =================================================================

        # Visualization mode: 'velocity', 'vorticity' (default), or 'none'
        #plot_mode = 'vorticity'

        plt.ion()                        # interactive mode for live animation
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)

        colorbar = None

        def plot_velocity(ux, uy, step, U_inlet, i, Re, Nx, Ny, obstacle):
            """Plot the velocity magnitude field |u| = sqrt(ux² + uy²)."""
            nonlocal colorbar
            speed = np.sqrt(ux**2 + uy**2)
            speed[obstacle] = np.nan      # mask cylinder

            ax.clear()
            im = ax.imshow(speed.T, origin='lower', cmap='viridis',
                    vmin=0, vmax=U_inlet * 2.0, aspect='auto',
                    extent=[0, Nx, 0, Ny])
            if colorbar is None or colorbar.ax is None:
                colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='|u|')
            else:
                colorbar.update_normal(im)
                colorbar.set_label('|u|')
            #ax.set_title(f"Velocity magnitude — step {step}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.tight_layout()
            out_dir = f"Figures/LBM/Velocity/Run{i}"
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(f"{out_dir}/re{Re}_step{step}.png")
            plt.pause(0.01)

        def plot_vorticity(ux, uy, step, U_inlet, i, Re, Nx, Ny, obstacle):
            """
            Plot the vorticity field (curl of velocity).
            Vorticity = ∂uy/∂x - ∂ux/∂y  — highlights the alternating vortices
            in the Karman street much more vividly than velocity magnitude.
            """
            nonlocal colorbar
            vorticity = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)
                    - np.roll(ux, -1, axis=1) + np.roll(ux, 1, axis=1))
            vorticity[obstacle] = np.nan

            ax.clear()
            im = ax.imshow(vorticity.T, origin='lower', cmap='plasma',
                    vmin=-0.04, vmax=0.04, aspect='auto',
                    extent=[0, Nx, 0, Ny])
            if colorbar is None or colorbar.ax is None:
                colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='vorticity')
            else:
                colorbar.update_normal(im)
                colorbar.set_label('vorticity')
            #ax.set_title(f"Vorticity field — step {step}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.tight_layout()
            out_dir = f"Figures/LBM/Vorticity/Run{i}"
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(f"{out_dir}/re{Re}_step{step}.png")
            plt.pause(0.01)

        def plot_field(ux, uy, step, U_inlet, plot_mode, i, Re, Nx, Ny, obstacle):
            """Dispatch to the selected visualization mode."""
            if plot_mode == 'vorticity':
                plot_vorticity(ux, uy, step, U_inlet, i, Re, Nx, Ny, obstacle)
            elif plot_mode == 'velocity':
                plot_velocity(ux, uy, step, U_inlet, i, Re, Nx, Ny, obstacle)
            else:
                pass  # no plotting

        # =================================================================
        # 7.  Main Simulation Loop
        # =================================================================
        #
        #  The LBM algorithm follows the "collide-then-stream" pattern:
        #    1. Compute macroscopic quantities (rho, u) from current distributions
        #    2. Collision:  relax f toward equilibrium  →  f_out
        #    3. Bounce-back: at obstacle nodes, replace f_out with reflected f
        #    4. Streaming:  propagate f_out along lattice velocities  →  f
        #    5. Boundary conditions (inlet/outlet) applied to post-streaming f
        #
        #  np.roll provides periodic wrapping, which serves as the y-boundary
        #  condition (periodic in the vertical direction).

        n_flow = 3   # base flow-throughs (scaled by Re below)
        monitor_every = 1000
        U_list, Re_list, Nx_list, Ny_list, r_cyl_list, cx_cyl_list, cy_cyl_list = get_params(Re_targets, tau_nom=0.6, Ma_max=0.1)
        for i in range(len(Re_list)):
            U_inlet = U_list[i]
            Re = Re_list[i]
            Nx = Nx_list[i]
            Ny = Ny_list[i]
            r_cyl = r_cyl_list[i]
            cx_cyl = cx_cyl_list[i]
            cy_cyl = cy_cyl_list[i]
            D = 2 * r_cyl
            nu  = U_inlet * D / Re
            flow_time = int(Nx / U_inlet)
            # Low Re: steady flow, 1 flow-through enough. High Re: need 3+
            if Re < 100:
                n_ft = 1 
                plot_every = 5000
            else:
                n_ft = n_flow
                plot_every = flow_time
            n_steps = n_ft * flow_time
            x = np.arange(Nx)
            y = np.arange(Ny)
            X, Y = np.meshgrid(x, y, indexing='ij')    # X,Y have shape (Nx, Ny)
            obstacle = (X - cx_cyl)**2 + (Y - cy_cyl)**2 <= r_cyl**2                   
            print(f"Simulation parameters:")
            print(f"  Grid:      {Nx} x {Ny}")
            print(f"  Cylinder:  center=({cx_cyl},{cy_cyl}), r={r_cyl}, D={D}")
            print(f"  Re={Re},  U_inlet={U_inlet},  nu={nu:.6f},  tau={tau:.4f}")
            # =================================================================
            # 5a. Initialization
            # =================================================================
            

            # Start with uniform flow at inlet velocity everywhere
            rho_init = np.ones((Nx, Ny))
            ux_init  = np.full((Nx, Ny), U_inlet)
            uy_init  = np.zeros((Nx, Ny))

            # Small transverse perturbation to break symmetry and trigger vortex shedding
            uy_init += 0.001 * U_inlet * np.sin(2.0 * np.pi * Y / Ny)

            # Set velocity to zero inside the obstacle
            ux_init[obstacle] = 0.0
            uy_init[obstacle] = 0.0

            # Initialize distributions to equilibrium
            f = equilibrium(rho_init, ux_init, uy_init, Nx, Ny)
            f_out = np.empty_like(f)

            # Precompute Smagorinsky vectors
            cc_xx = (c[:, 0] * c[:, 0]).astype(np.float64)
            cc_xy = (c[:, 0] * c[:, 1]).astype(np.float64)
            cc_yy = (c[:, 1] * c[:, 1]).astype(np.float64)

            # JIT warmup
            stream(f, f_out, c)
            f = equilibrium(rho_init, ux_init, uy_init, Nx, Ny)

            print(f"\nRunning {n_steps} timesteps ...")
            for step in range(1, n_steps + 1):
                # -------------------------------------------------------------
                # 7a.  Macroscopic quantities
                # -------------------------------------------------------------
                rho = np.sum(f, axis=2)
                ux  = np.sum(f * c[:, 0], axis=2) / rho
                uy  = np.sum(f * c[:, 1], axis=2) / rho

                # -------------------------------------------------------------
                # 7b.  Collision (BGK + Smagorinsky subgrid model)
                # -------------------------------------------------------------
                feq   = equilibrium(rho, ux, uy, Nx, Ny)
                f_neq = f - feq

                # Smagorinsky: adapt tau locally based on strain rate
                Qxx = f_neq @ cc_xx
                Qxy = f_neq @ cc_xy
                Qyy = f_neq @ cc_yy
                S_bar = np.sqrt(2.0 * (Qxx**2 + 2.0 * Qxy**2 + Qyy**2))
                tau_eff = 0.5 * (tau + np.sqrt(tau**2 + 18.0 * C_smag**2 * S_bar / rho))

                np.subtract(f, f_neq / tau_eff[..., np.newaxis], out=f_out)

                # -------------------------------------------------------------
                # 7c.  Bounce-back on obstacle
                # -------------------------------------------------------------
                f_out[obstacle] = f[obstacle][:, opp]

                # -------------------------------------------------------------
                # 7d.  Streaming (numba-accelerated, parallel)
                # -------------------------------------------------------------
                stream(f_out, f, c)

                # No-slip walls (top & bottom)
                f[:, 0,  2] = f[:, 0,  4]
                f[:, 0,  5] = f[:, 0,  7]
                f[:, 0,  6] = f[:, 0,  8]
                f[:, -1, 4] = f[:, -1, 2]
                f[:, -1, 7] = f[:, -1, 5]
                f[:, -1, 8] = f[:, -1, 6]

                # -------------------------------------------------------------
                # 7e.  Outlet boundary condition (zero-gradient / open)
                #      Copy from second-to-last column so vortices can leave.
                # -------------------------------------------------------------
                f[-1, :, :] = f[-2, :, :]

                # -------------------------------------------------------------
                # 7f.  Inlet boundary condition (Zou-He, fixed velocity)
                #      After streaming, populations 1, 5, 8 at x=0 are unknown
                #      (they would come from outside the domain).  Zou-He
                #      determines them from known populations and prescribed
                #      inlet velocity (ux=U_inlet, uy=0).
                # -------------------------------------------------------------
                rho_in = (  (f[0, :, 0] + f[0, :, 2] + f[0, :, 4])
                        + 2.0 * (f[0, :, 3] + f[0, :, 6] + f[0, :, 7])
                        ) / (1.0 - U_inlet)

                f[0, :, 1] = f[0, :, 3] + (2.0/3.0) * rho_in * U_inlet
                f[0, :, 5] = (f[0, :, 7]
                            - 0.5 * (f[0, :, 2] - f[0, :, 4])
                            + (1.0/6.0) * rho_in * U_inlet)
                f[0, :, 8] = (f[0, :, 6]
                            + 0.5 * (f[0, :, 2] - f[0, :, 4])
                            + (1.0/6.0) * rho_in * U_inlet)

                # -------------------------------------------------------------
                # 7g.  Visualization & progress
                # -------------------------------------------------------------
                if step % plot_every == 0:
                    plot_field(ux, uy, step, U_inlet,'vorticity', i, Re, Nx, Ny, obstacle)
                if step % plot_every == 0:
                    plot_field(ux, uy, step, U_inlet,'velocity', i, Re, Nx, Ny, obstacle)

                if step % 1000 == 0:
                    avg_rho = np.mean(rho[~obstacle])
                    print(f"  Step {step:>6d}/{n_steps}  |  avg density = {avg_rho:.6f}")

                if step % monitor_every == 0:
                    u_mag = np.sqrt(ux**2 + uy**2)
                    print(f"step {step:6d} | rho[{rho.min():.4f}, {rho.max():.4f}] | umax {u_mag.max():.4f}")
                    if not np.isfinite(f).all():
                        print(f"The simulation for Re = {Re} is no longer stable, stopping")
                        break


            # =================================================================
            # 8.  Final Output
            # =================================================================
            print("\nSimulation complete.")
            plt.ioff()
            #plt.savefig(f"Figures/LBM/re{Re}")
            plt.show()

    main()

