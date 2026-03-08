from asignment_2_1 import parallel_experiment, test_benchmark
import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame as df
import numpy as np
from assignment_2_2 import run_2_2
from assignment_2_3 import run_2_3
from plot import plot_dla_results

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
    print("Finished experiments and saved results to CSV files.")

    df_result = pd.DataFrame(results)
    df_result.to_csv("Assignment2/Figures/2.1/summary/dla_results.csv", index=False)
    
    non_parallel_time, parallel_time, grid_sizes = test_benchmark()
    plt.plot(grid_sizes, non_parallel_time, color='red', label='normal SOR run')
    plt.plot(grid_sizes, parallel_time, color='blue', label='parallelized SOR run')
    plt.xlabel('grid size')
    plt.ylabel('time (s)')
    plt.legend()
    plt.savefig('Figures/2.1/benchmark_test.png')
    plt.show()

    df = pd.DataFrame({
            "Grid sizes": grid_sizes,
            "Normal SOR run time means": non_parallel_time,
            "Parallel SOR run time means": parallel_time
        })
    df.to_csv("Figures/2.1/benchmark_test.csv", index=False)
    print('Finished running the benchmark test')

    plot_dla_results()
    print('Finished plotting the results of 2.1')

    
    run_2_2()
    print('Finished running 2.2')

    run_2_3()
    print('Finished running 2.3')
