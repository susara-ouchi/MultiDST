import numpy as np

def total_variation(ws):
    """
    Calculate the total variation of a weight vector.

    Parameters:
        ws (list or numpy array): Weight vector.

    Returns:
        float: Total variation of the weight vector.
    """
    return np.sum(np.abs(np.diff(ws)))

def uniform_deviation(ws):
    """
    Calculate the uniform deviation of a weight vector assuming uniform weights.

    Parameters:
        ws (list or numpy array): Weight vector.

    Returns:
        float: Uniform deviation of the weight vector.
    """
    return np.sum(np.abs(ws - 1))

import numpy as np

def thresholds_to_weights(ts, m_groups):
    """
    Convert thresholds to weights considering the number of items in each group.

    Parameters:
        ts (list or numpy array): Thresholds.
        m_groups (list or numpy array): Number of items in each group.

    Returns:
        numpy array: Weights.
    """
    ts = np.array(ts)  # Convert to NumPy array
    m_groups = np.array(m_groups)  # Convert to NumPy array

    if len(ts) != len(m_groups):
        raise ValueError("Inconsistent number of elements for thresholds and m_groups")

    nbins = len(ts)
    m = np.sum(m_groups)

    if np.all(ts == 0):
        return np.ones(nbins)
    else:
        return ts * m / np.sum(m_groups * ts)

def thresholds_to_weights_full(ts):
    """
    Convert thresholds to weights assuming equal group sizes.

    Parameters:
        ts (list or numpy array): Thresholds.

    Returns:
        numpy array: Weights.
    """
    m = len(ts)

    if np.all(ts == 0):
        return np.ones(m)
    else:
        return ts * m / np.sum(ts)

def normalize_weights(ws):
    """
    Normalize weights so that their sum will be equal to the length of the weight vector.
    Also ensure no negative weights appear.

    Parameters:
        ws (list or numpy array): Weight vector.

    Returns:
        numpy array: Normalized weights.
    """
    return np.abs(ws) * len(ws) / np.sum(np.abs(ws))

def regularize_weights(ws, lam):
    """
    Regularize weights by applying a linear combination of the original weights and their mean.

    Parameters:
        ws (list or numpy array): Weight vector.
        lam (float): Regularization factor (between 0 and 1).

    Returns:
        numpy array: Regularized weights.
    """
    if lam < 0 or lam > 1:
        raise ValueError("Regularization factor lambda should be in the interval [0,1]")

    ws = normalize_weights(ws)
    return (1 - lam) * ws + lam * np.mean(ws)

# Example usage:
weights = [0.1, 0.2, 0.3, 0.4]

# Calculate total variation
tv = total_variation(weights)
print("Total variation:", tv)

# Calculate uniform deviation
ud = uniform_deviation(weights)
print("Uniform deviation:", ud)

# Convert thresholds to weights
thresholds = [0.2, 0.3, 0.5]
m_groups = [100, 150, 200]
converted_weights = thresholds_to_weights(thresholds, m_groups)
print("Converted weights:", converted_weights)

# Normalize weights
normalized_weights = normalize_weights(weights)
print("Normalized weights:", normalized_weights)

# Regularize weights
regularized_weights = regularize_weights(weights, 0.5)
print("Regularized weights:", regularized_weights)
