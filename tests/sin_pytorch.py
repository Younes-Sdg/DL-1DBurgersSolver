import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the mathematical function
def func(x):
    return np.sin(x)

# Generate synthetic data
x_train = np.linspace(0, 2*np.pi, 100)
y_train = func(x_train)

# Define a simple neural network model
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

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Define hyperparameters
input_size = 1
hidden_size = 32
output_size = 1
learning_rate = 0.07
num_epochs = 1000

# Initialize the model, loss function, and optimizer
model = FunctionApproximator(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Set up the figure and axis for animation
fig, ax = plt.subplots()
ax.plot(x_train, func(x_train), label='Original Function')
line, = ax.plot(x_train, np.zeros_like(x_train), label='Neural Network Approximation', linestyle='--')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Function Approximation using Neural Network')
ax.grid(True)

# Animation loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        y_pred_tensor = model(x_train_tensor)
        y_pred = y_pred_tensor.detach().numpy()
        line.set_ydata(y_pred)
        ax.set_title(f'Epoch: {epoch+1}, Loss: {loss.item():.6f}')
        plt.pause(0.01)

plt.show()

img = mpimg.imread(r"D:\DL\DL-1DBurgersSolver\data\images\neural_net_1.png")

# Display the image
plt.imshow(img)
plt.axis('off')  # Turn off axis labels
plt.show()