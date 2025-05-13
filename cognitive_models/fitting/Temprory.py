import torch
import torch.optim as optim

# Define the function (unknown closed-form), but we can compute its stochastic gradient
# Example: A simple quadratic function with noise
def stochastic_gradient(x):
    true_grad = 2 * x  # True gradient of f(x) = x^2
    noise = torch.randn_like(x) * 0.1  # Add noise to simulate stochasticity
    return true_grad + noise

# Initialize the parameter to optimize
x = torch.tensor([5.0], requires_grad=True)  # Start at x=5

# Set up the Adam optimizer
optimizer = optim.Adam([x], lr=0.1)

# Optimization loop
num_iterations = 100
for i in range(num_iterations):
    optimizer.zero_grad()  # Reset gradients
    grad = stochastic_gradient(x)  # Get the stochastic gradient
    x.grad = grad  # Manually assign the gradient
    optimizer.step()  # Perform optimization step
    
    if (i + 1) % 10 == 0:
        print(f"Iteration {i+1}: x = {x.item():.4f}")

print("Optimization finished! Final value of x:", x.item())