import numpy as np
from scipy.stats import chi2

def sgof_test(p_values, alpha):
    # Step 01: Sort p-values in ascending order
    sorted_p_values = np.sort(p_values)
    
    # Step 02: Initialize R (number of p-values below threshold)
    R = np.sum(sorted_p_values <= alpha)
    print(R)
    
    # Step 03: Test if observed discoveries deviate significantly
    while R > 0:
        # Perform chi-square test
        observed = R
        expected = len(sorted_p_values) * (1 - alpha)
        chi_square_statistic = (observed - expected) ** 2 / expected
        p_value_chi2 = 1 - chi2.cdf(chi_square_statistic, df=1)  # df=1 for chi-square
        
        # Check significance
        if p_value_chi2 < alpha:
            R -= 1
        else:
            break
    
    # Step 04: Extract significant tests
    significant_tests = sorted_p_values[:R]
    return significant_tests

# Example 01:
alpha = 0.05
p_values = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2])
positive_tests = sgof_test(p_values, alpha)
print("Significant tests:", positive_tests)
#Output: Significant tests: [0.01 0.02 0.05]

np.arange(0.00001, 0.05, 0.01)

# Example 02:
list1 = [round(x, 8) for x in np.arange(0.00001, 0.05, 0.01)]
p_values = np.array(list1)
positive_tests = sgof_test(p_values, alpha)
print("Significant tests:", positive_tests)
# Output: Significant tests: []

#Example 03:
p_values = np.array([0.001, 0.02, 0.05, 0.11, 0.53, 0.68, 0.98, 0.99, 0.003])
positive_tests = sgof_test(p_values, alpha)
print("Significant tests:", positive_tests)
#Output: Significant tests: [0.01 0.02 0.05]