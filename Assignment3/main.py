from challenge_A_FEM import run_simulation
from finite_difference_navier_stokes import runallformain
import csv
import os
from multiprocessing import Pool

runs = [
    {"Re": 100,  "tau": 0.001,   "tend": 10.0},
    {"Re": 200,  "tau": 0.0005,  "tend": 10.0},
    {"Re": 300,  "tau": 0.0003,  "tend": 10.0},
    {"Re": 400,  "tau": 0.00025, "tend": 10.0},
    {"Re": 500,  "tau": 0.0002,  "tend": 10.0},
    {"Re": 550,  "tau": 0.00015, "tend": 10.0},
    {"Re": 750,  "tau": 0.00015, "tend": 10.0},
    {"Re": 1000, "tau": 0.0001,  "tend": 10.0},
]


if __name__ == "__main__":
    results = []
    for r in runs:
        print(f"Running Re = {r['Re']} tau={r['tau']}")
        
        res = run_simulation(**r)
        results.append((r, res))

    print("\n FEM Re sweep summary")
    print(f"{'Re':>6}  {'tau':>8}  {'stable':>7}  {'C_D':>8}  {'C_L_max':>8}  {'t_final':>8}")
    print("-" * 60)

    for r, res in results:
        stable = "NO" if res["diverged"] else "yes"
        cd = f"{res['C_D_mean']:.4f}" if res["C_D_mean"] is not None else "—"
        cl = f"{res['C_L_max']:.4f}" if res["C_L_max"] is not None else "—"
        print(f"{r['Re']:>6}  {r['tau']:>8.5f}  {stable:>7}  {cd:>8}  {cl:>8}  {res['t_final']:>8.4f}")

    folder = "ScientificComputing/Assignment3/Figures/FEM"
    os.makedirs(folder, exist_ok=True)
    csv_file =  os.path.join(folder, "summary_FEM.csv")

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Re", "tau", "stable", "C_D_mean", "C_L_max", "t_final"])
        writer.writeheader()
        for r, res in results:
            writer.writerow({
                "Re": r["Re"],
                "tau": r["tau"],
                "stable": "NO" if res["diverged"] else "yes",
                "C_D_mean": f"{res['C_D_mean']:.4f}" if res["C_D_mean"] is not None else "",
                "C_L_max": f"{res['C_L_max']:.4f}" if res["C_L_max"] is not None else "",
                "t_final": f"{res['t_final']:.4f}",
            })

    print(f"\nSaved summary -> {csv_file}")


    # (FD Navier Stokes)
    runallformain()


