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
from models.pinnet import PINN

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training will use: {device}")


model = PINN(input_dim=2, hidden_dim=50, layers=3).to(device)
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

        x_t_u0 = torch.cat((x, t, u0), dim=1).to(device)

        optimizer.zero_grad()
        loss = model.loss(x, t, u0)

        loss.backward()
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
line2, = ax.plot(x_numerical, np.zeros_like(x_numerical), 'b--', label='PINN Solution')
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
    x_t_u0 = torch.cat((x_tensor, t_tensor), 1).to(device)
    with torch.no_grad():
        u_pinn = model(x_t_u0).cpu().numpy()

    line1.set_ydata(u_numerical)
    line2.set_ydata(u_pinn.flatten())
    return line1, line2,

ani = FuncAnimation(fig, animate, frames=len(time_steps), interval=50, blit=True, repeat=False)
plt.show()
