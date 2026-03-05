import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame as df

# Plot errorbars
results_df = pd.read_csv("Assignment2/Figures/2.1/summary/dla_results.csv")
grouped = results_df.groupby(["eta", "omega"])
iterations = grouped["sor_iterations"]
summary = iterations.agg(["mean", "std"])
summary = summary.reset_index()
summary = summary.rename(columns={"mean": "average_iterations", "std": "std_iterations"})
eta_values = sorted(summary["eta"].unique())
plt.figure()

for eta in eta_values:
    eta_rows = summary[summary["eta"] == eta]
    sub = eta_rows.sort_values("omega")
    plt.errorbar(
        sub["omega"],
        sub["average_iterations"],
        yerr=sub["std_iterations"],
        marker="o",
        capsize=4,
        label=f"eta={eta}"
    )

plt.xlabel("omega")
plt.ylabel("Average SOR iterations")
plt.title("Average SOR iterations vs omega")
plt.legend()
plt.grid(True)

plt.savefig("Assignment2/Figures/2.1/summary/avg_iters_errorbars.png", dpi=150, bbox_inches="tight")
plt.savefig("Assignment2/Figures/2.1/summary/avg_iters_errorbars.pdf", dpi=150, bbox_inches="tight")