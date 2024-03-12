import numpy as np

def custom_transform(x, k=1.0):
    """
    Custom transformation function that shrinks coefficients towards zero.
    Args:
        x (float): Original coefficient value.
        k (float): Positive constant controlling the strength of the transformation.
    Returns:
        float: Transformed coefficient value.
    """
    return x / (1 + k * np.abs(x - 0.2))

# Example coefficients
coefficients = [0.2, 0.5, 0.6, 0.9, 0.02]

# Apply the custom transformation
transformed_coefficients = [custom_transform(x) for x in coefficients]

# Print the transformed coefficients
for i, coeff in enumerate(transformed_coefficients):
    print(f"Transformed coefficient for {coefficients[i]:.2f}: {coeff:.4f}")

# Feel free to experiment with different values of k to achieve the desired effect
