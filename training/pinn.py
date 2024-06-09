import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
from tqdm import tqdm
from models.pinnet import PINN
import time

# Setup the device for training (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training will use: {device}")

# Initialize the PINN model
model = PINN(input_dim=2, hidden_dim=50, layers=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

# Load real solution data
data = pd.read_csv('simulation_data.csv')
x_real = torch.tensor(data['x'].values, dtype=torch.float32).unsqueeze(1).to(device)
t_real = torch.tensor(data['time'].values, dtype=torch.float32).unsqueeze(1).to(device)
u_real = torch.tensor(data['u'].values, dtype=torch.float32).unsqueeze(1).to(device)

# Combine x and t into a single input tensor
inputs = torch.cat((x_real, t_real), dim=1)

# Training loop
def train_model(model, epochs, inputs, u_real, num_points, alpha, beta):
    model.train()
    best_loss = float('inf')
    best_model_state = None

    total_points = inputs.size(0)
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        # Randomly select a subset of points
        indices = torch.randperm(total_points)[:num_points]
        inputs_subset = inputs[indices]
        u_real_subset = u_real[indices]
        
        x = inputs_subset[:, 0].unsqueeze(1).to(device)
        t = inputs_subset[:, 1].unsqueeze(1).to(device)

        x.requires_grad_(True)
        t.requires_grad_(True)

        optimizer.zero_grad()
        loss = model.loss(x, t, u_real_subset, alpha, beta)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            tqdm.write(f'Epoch {epoch}: Loss {loss.item()}')

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()

    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')
    torch.save(best_model_state, 'trained_models/pinns_best.pth')

    # Save the best model parameters and training details
    params_path = 'trained_models/pinns_best_parameters.txt'
    with open(params_path, 'w') as f:
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Learning Rate: {0.001}\n")
        f.write(f"Weight Decay: {1e-4}\n")
        f.write(f"Step Size: {1000}\n")
        f.write(f"Input Dimension: {2}\n")
        f.write(f"Hidden Dimension: {50}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Alpha: {beta}\n")
        f.write(f"Best Loss: {best_loss:.6f}\n")
        

# Parameters for training
epochs = 15000
num_points = 100 # Number of points to randomly sample for each epoch
alpha = 0.3
beta = 0.7

# Calculate the training time
start_time = time.time()
train_model(model, epochs, inputs, u_real, num_points, alpha, beta)
end_time = time.time()
training_time_minutes = (end_time - start_time) / 60

# Load data for animation
x_numerical = data['x'].unique()
time_steps = data['time'].unique()

# Set up the plotting for animation
fig, ax = plt.subplots(figsize=(10, 6))
line1, = ax.plot(x_numerical, np.zeros_like(x_numerical), 'r-', label='Numerical Solution')
line2, = ax.plot(x_numerical, np.zeros_like(x_numerical), 'b--', label='PINN Solution')
ax.set_xlim(-10, 10)
ax.set_ylim(-0.3, 1.1)
ax.set_title(f"Burgers' Equation Solutions - Training Time: {training_time_minutes:.2f} minutes")
ax.set_xlabel('X')
ax.set_ylabel('U')
ax.legend()

def animate(i):
    """ Update function for animation """
    t = time_steps[i]
    u_numerical = data[data['time'] == t]['u'].values
    
    t_tensor = torch.full((len(x_numerical), 1), t, dtype=torch.float32).to(device)
    x_tensor = torch.Tensor(x_numerical).unsqueeze(1).to(device)
    x_t_u0 = torch.cat((x_tensor, t_tensor), 1).to(device)
    with torch.no_grad():
        u_pinn = model(x_t_u0).cpu().numpy()

    line1.set_ydata(u_numerical)
    line2.set_ydata(u_pinn.flatten())
    return line1, line2,

# Run the animation
ani = FuncAnimation(fig, animate, frames=len(time_steps), interval=50, blit=True, repeat=False)

# Save the animation as a GIF
output_path = os.path.join('animations', 'pinn_animation.gif')
ani.save(output_path, writer='pillow', fps=10)
print("Animation saved to:", output_path)

plt.show()
