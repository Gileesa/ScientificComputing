import os
import pandas as pd
import matplotlib.pyplot as plt

# Load data
csv_path = "Assignment2/Figures/2.1/summary/dla_results.csv"
outdir = "Assignment2/Figures/2.1/summary"
os.makedirs(outdir, exist_ok=True)

df = pd.read_csv(csv_path)

# Summary statistics
summary = (
    df.groupby(["eta", "omega"])["sor_iterations"]
    .agg(
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        count="count"
    )
    .reset_index()
)

eta_list = [0, 0.5, 1.0, 1.5, 2.0]
omega_list = [1.75, 1.8, 1.85, 1.9, 1.95]


summary = summary.sort_values(["eta", "omega"])
summary.to_csv(f"{outdir}/dla_summary_stats.csv", index=False)

print("\nSummary statistics:")
print(summary)

# Best omega per eta by mean
best_per_eta = summary.loc[summary.groupby("eta")["mean"].idxmin()].copy()
print("\nBest omega for each eta (by mean):")
print(best_per_eta[["eta", "omega", "mean", "std", "median"]])

# Boxplots
etas = sorted(df["eta"].unique())
omegas = sorted(df["omega"].unique())

fig, axes = plt.subplots(1, len(etas), figsize=(18, 4), sharey=True)

for ax, eta in zip(axes, etas):
    sub = df[df["eta"] == eta]
    data = [sub[sub["omega"] == om]["sor_iterations"] for om in omegas]

    ax.boxplot(data, tick_labels=[str(o) for o in omegas], showfliers=False)
    ax.set_title(rf"$\eta = {eta}$")
    ax.set_xlabel(r"$\omega$")

axes[0].set_ylabel("SOR iterations")
fig.suptitle("SOR iteration distributions for each eta")
fig.tight_layout()
fig.savefig(f"{outdir}/boxplot_by_eta.png", dpi=150)
fig.savefig(f"{outdir}/boxplot_by_eta.pdf")
plt.close(fig)

# Mean iterations vs omega 
plt.figure(figsize=(8, 5))

for eta in eta_list:
    sub = summary[summary["eta"] == eta]
    plt.plot(sub["omega"], sub["mean"], marker="o", label=rf"$\eta={eta}$")

plt.xticks(omega_list)
plt.xlabel(r"$\omega$")
plt.ylim(0, 100)
plt.ylabel("Mean SOR iterations")
plt.title("Mean SOR iterations vs omega")
plt.legend()
plt.tight_layout()
plt.savefig(f"{outdir}/mean_vs_omega_by_eta.png", dpi=150)
plt.savefig(f"{outdir}/mean_vs_omega_by_eta.pdf")
plt.close()

# median iterations vs omega 
plt.figure(figsize=(8, 5))

for eta in eta_list:
    sub = summary[summary["eta"] == eta]
    plt.plot(sub["omega"], sub["median"], marker="o", label=rf"$\eta={eta}$")

plt.xticks(omega_list)
plt.xlabel(r"$\omega$")
plt.ylabel("Median SOR iterations")
plt.ylim(0, 100)
plt.title("Median SOR iterations vs omega")
plt.legend()
plt.tight_layout()
plt.savefig(f"{outdir}/median_vs_omega_by_eta.png", dpi=150)
plt.savefig(f"{outdir}/median_vs_omega_by_eta.pdf")
plt.close()