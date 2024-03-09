import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from tabulate import tabulate  # Assuming this is used for formatting tables

# Import simulation functions and correction methods
from A01_sim_data import simulation_01
from A02_FWER1_bonferroni import bonferroni

# Function to evaluate simulation results
def sim_evaluation(adj_p, sig_index, fire_index, nonfire_index, p_values, threshold=0.05):
    """
    Evaluate simulation results.

    Args:
        adj_p (list): Adjusted p-values.
        sig_index (list): Indices of significant p-values.
        fire_index (list): Indices of p-values indicating firing.
        nonfire_index (list): Indices of p-values indicating non-firing.
        p_values (list): Original p-values.
        threshold (float): Significance threshold.

    Returns:
        tuple: Tuple containing significant p-values for firing and non-firing, counts of firing and non-firing p-values.

    """
    significant_p_fire = [index for index in sig_index if index in fire_index]
    significant_p_nonfire = [index for index in sig_index if index in nonfire_index]
    return significant_p_fire, significant_p_nonfire, len(fire_index), len(nonfire_index)

# Function to run simulations and compute performance metrics
def power_simulation(num_simulations, num_firing, num_nonfire, effect, pi0,  n0, n1, s0, s1):
    """
    Run simulations and compute performance metrics.

    Args:
        num_simulations (int): Number of simulations to run.
        num_firing (int): Number of firing samples.
        num_nonfire (int): Number of non-firing samples.
        effect (float): Effect size.
        n0 (int): Sample size for group 0.
        n1 (int): Sample size for group 1.
        s0 (float): Variability parameter for group 0.
        s1 (float): Variability parameter for group 1.

    Returns:
        tuple: Tuple containing mean and standard deviation of power, FDR, accuracy, FPR, and F1 score.

    """
    sim_power = []
    sim_acc = []
    sim_fdr = []
    sim_fpr = []
    sim_f1 = []

    for _ in range(num_simulations):
        sim_data = simulation_01(seed=None, num_firing=num_firing, num_nonfire=num_nonfire, effect=effect,
                                 n0=n0, n1=n1, threshold=0.05, show_plot=False, s0=s0, s1=s1)
        p_values, significant_p, fire_index, nonfire_index = sim_data[0], sim_data[1], sim_data[2], sim_data[3]
        adj_p = bonferroni(p_values, alpha=0.05, weights=False)[0]
        significant_p = bonferroni(p_values, alpha=0.05, weights=False)[1]
        sig_fire, sig_nonfire, p_fire, p_nonfire = sim_evaluation(adj_p, significant_p, fire_index, nonfire_index,
                                                                   p_values, threshold=0.05)

        # Calculate performance metrics
        TP = len(sig_fire)
        FP = len(sig_nonfire)
        TN = p_nonfire - FP
        FN = p_fire - TP
        sensitivity = TP / (TP + FN) if TP + FN != 0 else 0
        specificity = TN / (TN + FP) if TN + FP != 0 else 0
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if precision + sensitivity != 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2
        fpr = 1 - specificity
        fdr = FP / (TP + FP) if TP + FP != 0 else 0

        sim_power.append(sensitivity)
        sim_fdr.append(fdr)
        sim_acc.append(balanced_accuracy)
        sim_fpr.append(fpr)
        sim_f1.append(f1_score)

    # Compute mean and standard deviation of performance metrics
    power_mean = np.mean(sim_power)
    power_sd = np.std(sim_power)
    fdr_mean = np.mean(sim_fdr)
    fdr_sd = np.std(sim_fdr)
    acc_mean = np.mean(sim_acc)
    acc_sd = np.std(sim_acc)
    fpr_mean = np.mean(sim_fpr)
    fpr_sd = np.std(sim_fpr)
    f1_mean = np.mean(sim_f1)
    f1_sd = np.std(sim_f1)

    return power_mean, power_sd, fdr_mean, fdr_sd, acc_mean, acc_sd, fpr_mean, fpr_sd, f1_mean, f1_sd

# Function to run simulations for multiple parameter combinations
def run_simulations(num_simulations):
    """
    Run simulations for multiple parameter combinations.

    Args:
        num_simulations (int): Number of simulations to run.

    Returns:
        pd.DataFrame: DataFrame containing simulation results.

    """
    sample_sizes = [5, 15]  # Sample sizes from Kang et al.
    parameters = []

    # Generate combinations of parameters
    for n0 in sample_sizes:
        for n1 in sample_sizes:
            for num_firing in [1000, 5000]:  # From BonEV
                num_nonfire = 10000 - num_firing
                pi0 = num_nonfire / 10000
                for effect in [0.5, 1.0, 1.5]:  # From SGoF
                    for s0 in [0.5, 1]:
                        for s1 in [0.5, 1.0]:
                            parameters.append((n0, n1, num_firing, num_nonfire, pi0, effect, s0, s1))

    # Execute simulations in parallel
    results = Parallel(n_jobs=-1)(delayed(power_simulation)(*params, num_simulations) for params in parameters)

    # Process simulation results
    columns = ['n0', 'n1', 'num_firing', 'num_nonfire', 'pi0', 'effect', 's0', 's1', 'Power Mean', 'Power SD',
               'FDR Mean', 'FDR SD', 'Accuracy Mean', 'Accuracy SD', 'FPR Mean', 'FPR SD', 'F1 Mean', 'F1 SD']
    df_results = pd.DataFrame(results, columns=columns)

    return df_results

# Main function to run simulations and save results
def main():
    num_simulations = 1
    df_results = run_simulations(num_simulations)
    print(df_results)
    # Save results to CSV
    df_results.to_csv('simulation_results.csv', index=False)

if __name__ == "__main__":
    main()
