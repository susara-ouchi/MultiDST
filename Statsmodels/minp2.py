import numpy as np

def minP_permutation_test(p_values, num_permutations=10000):
    # Sort p-values in ascending order
    sorted_p_values = np.sort(p_values)

    # Initialize an array to store adjusted p-values
    adjusted_p_values = np.zeros(len(sorted_p_values))

    # Permutation procedure
    for i, p_val in enumerate(sorted_p_values):
        # Simulate num_permutations random permutations
        permuted_stats = []
        for _ in range(num_permutations):
            # Shuffle the group labels (assuming two groups)
            shuffled_labels = np.random.permutation([0] * len(p_values))
            permuted_data = np.array(p_values)[shuffled_labels == 0]

            # Compute the permutation test statistic (e.g., mean difference)
            permuted_stat = np.abs(np.mean(permuted_data) - np.mean(p_values))
            permuted_stats.append(permuted_stat)

        # Calculate the adjusted p-value
        adjusted_p_values[i] = np.mean(permuted_stats <= np.abs(np.mean(p_values) - np.mean(permuted_data)))

    # Enforce monotonicity
    for j in range(len(adjusted_p_values) - 1):
        adjusted_p_values[j] = max(adjusted_p_values[j], adjusted_p_values[j + 1])

    return adjusted_p_values

# Example:
input_p_values = [0.001, 0.012, 0.038, 0.14, 0.25, 0.31, 0.55, 0.99]  
adjusted_p_vals = minP_permutation_test(input_p_values)

print("Adjusted p-values:", adjusted_p_vals)
