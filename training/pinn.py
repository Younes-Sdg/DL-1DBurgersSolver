import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
from tqdm import tqdm
from data_generator import functions
from models.pinnet import PINN

# Setup the device for training (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training will use: {device}")

# Initialize the PINN model
model = PINN(input_dim=2, hidden_dim=50, layers=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

def train_model(model, epochs, domain, time_frame):
    """
    Inner function to train the model.

    Args:
    model (nn.Module): The neural network model to train.
    epochs (int): Number of training epochs.
    domain (list): Spatial domain as [min, max].
    time_frame (list): Time frame as [start, end].

    Returns:
    float: The loss from the last training epoch.
    """
    model.train()
    best_loss = float('inf')
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        x = torch.rand(1000, 1) * (domain[1] - domain[0]) + domain[0]
        t = torch.linspace(time_frame[0], time_frame[1], 1000).unsqueeze(1)
        u0 = torch.tensor([functions.initial_condition(xi.item()) for xi in x], dtype=torch.float32).unsqueeze(1)

        x.requires_grad_(True)
        t.requires_grad_(True)

        optimizer.zero_grad()
        loss = model.loss(x, t, u0)
        loss.backward()
        optimizer.step()
        scheduler.step()
            
    return loss.item()

# Parameters for training
epochs = 5000
space_domain = [-10, 10]
time_domain = [0, 5]

# Training the model
loss = train_model(model, epochs, space_domain, time_domain)
print(f'Final training loss = {loss}')

# Load data for animation
data = pd.read_csv('simulation_data.csv')
x_numerical = data['x'].unique()
time_steps = data['time'].unique()

# Set up the plotting for animation
fig, ax = plt.subplots(figsize=(10, 6))
line1, = ax.plot(x_numerical, np.zeros_like(x_numerical), 'r-', label='Numerical Solution')
line2, = ax.plot(x_numerical, np.zeros_like(x_numerical), 'b--', label='PINN Solution')
ax.set_xlim(-10, 10)
ax.set_ylim(-0.3, 1.1)
ax.set_title("Burgers' Equation Solutions")
ax.set_xlabel('X')
ax.set_ylabel('U')
ax.legend()

mse_list = []

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
    mse = np.mean((u_numerical - u_pinn.flatten())**2)
    mse_list.append(mse)
    
    return line1, line2,

# Run the animation
ani = FuncAnimation(fig, animate, frames=len(time_steps), interval=50, blit=True, repeat=False)

# Save the animation as a GIF
output_path = os.path.join('animations', 'pinn_animation.gif')
ani.save(output_path, writer='pillow', fps=10)
print("Animation saved to:", output_path)

plt.show()