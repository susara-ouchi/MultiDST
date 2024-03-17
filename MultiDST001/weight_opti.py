import numpy as np
import scipy.optimize as opt

def my_function(params):
    param1, param2, param3, param4, param5, param6, param7 = params
    return param3

# Initial guess for parameters
initial_params = [1/2, 0.4, 1, 5, 6, 2, 1]

# Bounds for each parameter
bounds = [(0, 0.5), (0, 0.5), (0, 1), (1, 2), (1, 2), (2, 3), (1, 1)]

# Constraint: Sum of parameters = 10 (you can adjust this value)
constraint_matrix = np.array([[1, 1, 1, 1, 1, 1, 1]])
constraint_rhs = np.array([100])  # Adjust the sum value as needed

# Create the LinearConstraint
sum_constraint = opt.LinearConstraint(constraint_matrix, lb=constraint_rhs, ub=constraint_rhs)

# Minimize the function with the constraint
result = opt.minimize(my_function, initial_params, bounds=bounds, constraints=sum_constraint)
optimized_params = result.x
print("Optimized parameters:", optimized_params)
