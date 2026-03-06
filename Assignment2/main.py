from asignment_2_1 import parallel_experiment
import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame as df
import numpy as np

if __name__ == "__main__":
    eta_list = [0, 0.5, 1.0, 1.5, 2.0]
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
    os.makedirs("Assignment2/Figures/2.1/summary", exist_ok=True)

    df_result = pd.DataFrame(results)
    df_result.to_csv("Assignment2/Figures/2.1/summary/dla_results.csv", index=False)
    print("Finished experiments and saved results to CSV files.")