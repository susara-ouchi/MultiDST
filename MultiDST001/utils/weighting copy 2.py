from scipy.optimize import minimize

# Define the objective function
def objective_function(x):
    return x[0]**2 + x[1]**2

# Define the inequality constraint: x + y >= 1
def inequality_constraint(x):
    return x[0] + x[1] - 1

# Define the non-negativity constraints: x >= 0, y >= 0
def non_negativity_constraint(x):
    return x

# Define the initial guess
initial_guess = [0.5, 0.5]

# Define the bounds for each variable (x, y)
bounds = [(0, None), (0, None)]

# Define the constraints
constraints = [{'type': 'ineq', 'fun': inequality_constraint},
               {'type': 'ineq', 'fun': non_negativity_constraint}]

# Optimize the function
optimization_result = minimize(objective_function, initial_guess, bounds=bounds, constraints=constraints)

# Extract the optimal solution
optimal_solution = optimization_result.x

# Print the optimal solution
print("Optimal solution:", optimal_solution)

# Print the value of the objective function at the optimal solution
print("Minimum value of the objective function:", optimization_result.fun)
