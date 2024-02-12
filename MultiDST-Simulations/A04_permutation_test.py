import numpy as np

from A01_sim_data import p_values, fire_index, nonfire_index
from A01_weighting import weighted_p

def permutation_adj(p_values, num_permutations):
    # Number of tests
    num_tests = len(p_values)
    # To store the number of times the permuted p-values are less than the original p-values
    permuted_counts = np.zeros(num_tests)
    
    for i in range(num_permutations):
        # Permute the p-values
        permuted_p_values = np.random.permutation(p_values)
        # Count the number of permuted p-values less than the original p-values
        permuted_counts += (permuted_p_values < p_values)
    
    # Calculate corrected p-values using Bonferroni method
    corrected_p_values = permuted_counts / num_permutations
    return corrected_p_values

def permutation_test(p_values,num_permutations, alpha = 0.05, weights = False):
    if weights == True:
        perm = permutation_adj(weighted_p, num_permutations)
        adj_p = list(perm)
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]
    else:
        perm = permutation_adj(p_values,num_permutations)
        adj_p = list(perm)
        sig_index = [index for index,p in enumerate(adj_p) if p < alpha]
    return adj_p,sig_index

perm_p = permutation_test(p_values,1000)[0]
perm_sig = permutation_test(p_values,1000)[1]
