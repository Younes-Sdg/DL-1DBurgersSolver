import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the mathematical function
def func(x):
    return np.sin(x)

# Define the derivative of the function
def func_derivative(x):
    return np.cos(x)

# Generate synthetic data with reduced number of data points for training
x_train_small = np.linspace(0, 2*np.pi, 20)  # Generating 20 data points for training
y_train_small = func(x_train_small)

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(x_train_small, dtype=torch.float32).view(-1, 1)
y_train_tensor = torch.tensor(y_train_small, dtype=torch.float32).view(-1, 1)

# Define a neural network model with two outputs: u(x) and u'(x)
class FunctionApproximator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FunctionApproximator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define hyperparameters
input_size = 1
hidden_size = 32
output_size = 2  # Two outputs: u(x) and u'(x)
learning_rate = 0.1
num_epochs = 1000

# Initialize the model, loss function, and optimizer
model = FunctionApproximator(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Set up the figure and axis for animation
fig, ax = plt.subplots()
x_values = np.linspace(0, 2*np.pi, 100)
ax.plot(x_values, func(x_values), label='Real Sin(x)')  # Plotting the real sine function
line, = ax.plot(x_values, np.zeros_like(x_values), label='Neural Network Approximation', linestyle='--')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Function Approximation using Neural Network')
ax.grid(True)

# Animation loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    
    # Split the outputs into u(x) and u'(x)
    u_values = outputs[:, 0].view(-1, 1)
    uprime_values = outputs[:, 1].view(-1, 1)
    
    # Loss for fitting the data
    loss_data = criterion(u_values, y_train_tensor)
    
    # Loss for enforcing the condition u'(x) = cos(x)
    loss_phys = criterion(uprime_values, torch.tensor(func_derivative(x_train_small), dtype=torch.float32).view(-1, 1))
    
    # Total loss
    loss = loss_data + loss_phys
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        y_pred_tensor = model(torch.tensor(x_values, dtype=torch.float32).view(-1, 1))
        y_pred = y_pred_tensor[:, 0].detach().numpy()  # Extracting u(x) for plotting
        line.set_ydata(y_pred)
        ax.set_title(f'Epoch: {epoch+1}, Loss: {loss.item():.6f}')
        plt.pause(0.01)

plt.show()
