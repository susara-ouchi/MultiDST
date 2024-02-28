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


# Import necessary libraries
import numpy as np
from scipy.stats import chi2, rankdata

def sgof_chi_square(p_values):
    # Sort the p-values in ascending order
    sorted_p_values = np.sort(p_values)

    # Calculate the empirical cumulative distribution function (CDF)
    empirical_cdf = np.arange(1, len(sorted_p_values) + 1) / len(sorted_p_values)

    # Calculate the Chi-Square statistic
    chi_square_statistic = np.sum((sorted_p_values - empirical_cdf)**2 / empirical_cdf)

    # Calculate the p-value of the Chi-Square test
    chi_square_p_value = 1 - chi2.cdf(chi_square_statistic, df=len(sorted_p_values) - 1)

    return chi_square_p_value

def benjamini_hochberg(p_values):
    # Calculate the ranks of the p-values
    ranks = rankdata(p_values)

    # Calculate the adjusted p-values using the Benjamini-Hochberg procedure
    adjusted_p_values = p_values * len(p_values) / ranks

    # Ensure that the adjusted p-values are between 0 and 1
    adjusted_p_values = np.minimum.accumulate(adjusted_p_values[::-1])[::-1]

    return adjusted_p_values

# Define your p-values
p_values = np.array([...])  # insert your p-values here

# Conduct the SGoF test using the Chi-Square test
sgof_p_value = sgof_chi_square(p_values)

# Adjust the p-values using the Benjamini-Hochberg procedure
adjusted_p_values = benjamini_hochberg(p_values)

print('SGoF p-value:', sgof_p_value)
print('Adjusted p-values:', adjusted_p_values)
