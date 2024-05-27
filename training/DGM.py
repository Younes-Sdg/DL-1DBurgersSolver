import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from tqdm import tqdm

from data_generator import functions
from models.dgmnet import DGMNet

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training will use: {device}")


model = DGMNet(input_dim=2, hidden_dim=50, layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

def train_model(model, epochs, domain, time_frame):
    model.train()
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        x = torch.rand(1000, 1) * (domain[1] - domain[0]) + domain[0]
        t = torch.rand(1000, 1) * (time_frame[1] - time_frame[0]) + time_frame[0]

        u0 = torch.tensor([functions.initial_condition(xi.item()) for xi in x], dtype=torch.float32).unsqueeze(1)

        x.requires_grad_(True)
        t.requires_grad_(True)

        x_t = torch.cat((x, t), dim=1).to(device)

        optimizer.zero_grad()
        u_pred = model(x_t)

        grad_outputs = torch.ones_like(u_pred)
        u_t = torch.autograd.grad(u_pred, t, grad_outputs=grad_outputs, create_graph=True)[0]
        u_x = torch.autograd.grad(u_pred, x, grad_outputs=grad_outputs, create_graph=True)[0]

        # Calculate residuals
        f = u_t + u_pred * u_x

        # Initial condition
        ic = torch.tensor([functions.initial_condition(xi.item()) for xi in x], dtype=torch.float32).to(device)
        loss_ic = ((u_pred.squeeze() - ic) ** 2).mean()

        # Boundary conditions
        x_boundary_left = torch.full((1000, 1), domain[0]).to(device)
        x_boundary_right = torch.full((1000, 1), domain[1]).to(device)
        t_boundary = (torch.rand(1000, 1) * (time_frame[1] - time_frame[0]) + time_frame[0]).to(device)
        u_boundary_left = model(torch.cat((x_boundary_left, t_boundary), dim=1)).squeeze()
        u_boundary_right = model(torch.cat((x_boundary_right, t_boundary), dim=1)).squeeze()

        # Updated boundary conditions to 0.2 and 0.4
        loss_bc = ((u_boundary_left - 0.2) ** 2).mean() + ((u_boundary_right - 0.4) ** 2).mean()

        # Combine losses
        loss_pde = (f ** 2).mean()
        loss = loss_pde + loss_ic + loss_bc

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            tqdm.write(f'Epoch {epoch}: Loss {loss.item()}')



epochs = 1500
space_domain = [-10, 10]
time_domain = [0,5]

train_model(model, epochs, space_domain , time_domain )

data = pd.read_csv('simulation_data.csv')
x_numerical = data['x'].unique()
time_steps = data['time'].unique()

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

ani = FuncAnimation(fig, animate, frames=len(time_steps), interval=50, blit=True, repeat=False)
plt.show()
