import numpy as np
from scipy.stats import chi2

def SGoF(p_values, alpha):
    # Count the number of significant discoveries
    R = np.sum(np.array(p_values) < alpha)
    m = len(p_values)
    
    # Step 3: Repeat until there are no more p-values to examine
    while m > 0:
        # Step 3a: Check for deviations using chi-square test
        expected_F = m * alpha
        observed_F = R
        chi_squared_stat = (observed_F - expected_F)**2 / expected_F
        
        # Approximate chi-square test with degrees of freedom = 1
        p_value_chi2 = 1 - chi2.cdf(chi_squared_stat, df=1)
        
        # If chi-square test is significant
        if p_value_chi2 < alpha:
            # Add one more significant discovery
            R += 1
            # Update counts
            m -= 1  # Decrement m since we have one less test to consider
        else:
            # Step 3b: If chi-square test is not significant, stop the process
            break
    
    # Step 4: Output the number of significant discoveries
    return R

# Example usage:
p_values = [0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
alpha = 0.05
num_significants = SGoF(p_values, alpha)
print("Number of significant discoveries:", num_significants)
