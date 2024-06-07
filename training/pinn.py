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
from models.pinnet import train_model,setup_animation

# Setup the device for training (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training will use: {device}")

# Initialize the PINN model
model = PINN(input_dim=2, hidden_dim=50, layers=8).to(device)

# Parameters for training
epochs = 1000
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
fig, ax, line1, line2 = setup_animation(data, x_numerical, time_steps)

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
plt.show()

# Calculate the maximum MSE after the animation
mse_max = max(mse_list) if mse_list else None

# Save the animation
writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
ani.save(f'animation_pinns.mp4', writer=writer)

output_file = 'training/animation_pinns.mp4'
ani.save(output_file, writer=writer)
print(f"Animation saved as {output_file}")