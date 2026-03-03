from asignment_2_1 import parallel_experiment
import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame as df
import numpy as np

if __name__ == "__main__":
    eta_list = [0.5, 1.0, 1.5, 2.0]
    omega_list = [1.75, 1.8, 1.85, 1.9, 1.95]

    results = parallel_experiment(
        eta_list = eta_list, 
        omega_list = omega_list,
        steps=1000,
        grid_size=100,
        seed=0,
        progress_every=50,
        max_sor_iterations=1000,
        interval=50,
        tail=50,
        workers=None
        )

    # Save results to a CSV file
    os.makedirs("Assignment2/2.1/grid", exist_ok=True)

    results_df = pd.DataFrame(results, columns=["eta", "omega", "average_iterations", "max_iterations"])
    results_df.to_csv("Assignment2/2.1/grid/dla_results.csv", index=False)

    print("Finished running all results")

    
    plt.figure()

    eta_values = sorted(results_df["eta"].unique())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(eta_values)))

    for color, eta in zip(colors, eta_values):
        sub = results_df[results_df["eta"] == eta].sort_values("omega")
        plt.plot(
            sub["omega"], 
            sub["average_iterations"], 
            marker="o", 
            label=f"eta={eta}", 
            color=color)
    
    plt.xlabel("omega")
    plt.ylabel("Average SOR iterations")
    plt.title("Average SOR iterations vs omega (for each eta)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Figures/2.1/grid/curves_avg_iters.png", dpi=150, bbox_inches="tight")
    plt.savefig("Figures/2.1/grid/curves_avg_iters.pdf", dpi=150, bbox_inches="tight")
    plt.show()
