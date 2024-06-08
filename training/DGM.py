import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from models.dgmnet import DGMNet
from data_generator import functions
import os

# Setup device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training will use: {device}")

# Initialize the model
model = DGMNet(input_dim=2, hidden_dim=50, layers=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

def train_model(model, epochs, domain, time_frame, num_points):
    model.train()
    
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        x = torch.rand(num_points, 1) * (domain[1] - domain[0]) + domain[0]
        t = torch.linspace(time_frame[0], time_frame[1], 1000).unsqueeze(1)
        ic = torch.tensor([functions.initial_condition(xi.item()) for xi in x], dtype=torch.float32).unsqueeze(1).to(device)

        x.requires_grad_(True)
        t.requires_grad_(True)

        optimizer.zero_grad()
        loss = model.loss(x, t, ic)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss {loss.item()}')

# Training parameters
epochs = 5000   
space_domain = [-10, 10]
time_domain = [0, 5]
num_points = 1000

# Start training
train_model(model, epochs, space_domain, time_domain, num_points)

# Load data for visualization
data = pd.read_csv('simulation_data.csv')
x_numerical = data['x'].unique()
time_steps = data['time'].unique()

# Setup for animation
fig, ax = plt.subplots(figsize=(10, 6))
line1, = ax.plot(x_numerical, np.zeros_like(x_numerical), 'r-', label='Numerical Solution')
line2, = ax.plot(x_numerical, np.zeros_like(x_numerical), 'b--', label='DGM Solution')
ax.set_xlim(-10, 10)
ax.set_ylim(-0.3, 1.1)
ax.set_title("Burgers' Equation Solutions")
ax.set_xlabel('X')
ax.set_ylabel('U')
ax.legend()

def animate(i):
    t = time_steps[i]
    u_numerical = data[data['time'] == t]['u'].values

    t_tensor = torch.full((len(x_numerical), 1), t, dtype=torch.float32).to(device)
    x_tensor = torch.Tensor(x_numerical).unsqueeze(1).to(device)
    x_t = torch.cat((x_tensor, t_tensor), 1).to(device)
    with torch.no_grad():
        u_dgm = model(x_t).cpu().numpy()

    line1.set_ydata(u_numerical)
    line2.set_ydata(u_dgm.flatten())
    return line1, line2,

ani = FuncAnimation(fig, animate, frames=len(time_steps), interval=50, blit=True)

# Save the animation
output_path = os.path.join('animations', 'pinn_animation.gif')
ani.save(output_path, writer='pillow', fps=10)
print("Animation saved to:", output_path)

plt.show()
