import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import functions

class DGMNet(nn.Module):
    def __init__(self):
        super(DGMNet, self).__init__()
        self.fc1 = nn.Linear(2, 50)  # First hidden layer with 50 neurons
        self.fc2 = nn.Linear(50, 50) # Second hidden layer with 50 neurons
        self.fc3 = nn.Linear(50, 50) # Additional hidden layer
        self.fc4 = nn.Linear(50, 50) # Additional hidden layer
        self.fc5 = nn.Linear(50, 1)  # Output layer
        self.activation = nn.Tanh()  # Using Tanh activation function

    def forward(self, x_t):
        x = self.activation(self.fc1(x_t))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        return self.fc5(x)

model = DGMNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, epochs, domain, time_frame):
    model.train()
    for epoch in range(epochs):
        x = torch.rand(1000, 1) * (domain[1] - domain[0]) + domain[0]
        t = torch.rand(1000, 1) * (time_frame[1] - time_frame[0]) + time_frame[0]
        
        x.requires_grad_(True)
        t.requires_grad_(True)

        x_t = torch.cat((x, t), dim=1)

        optimizer.zero_grad()
        u_pred = model(x_t)

        grad_outputs = torch.ones_like(u_pred)
        u_t = torch.autograd.grad(u_pred, t, grad_outputs=grad_outputs, create_graph=True)[0]
        u_x = torch.autograd.grad(u_pred, x, grad_outputs=grad_outputs, create_graph=True)[0]

        f = u_t + u_pred * u_x  # Keeping the complex loss function

        ic = torch.tensor([functions.initial_condition(xi.item()) for xi in x], dtype=torch.float32)
        loss = (f ** 2).mean() + ((u_pred.squeeze() - ic) ** 2).mean()
        loss += ((model(torch.cat((torch.full_like(x, domain[0]), t), 1)).squeeze(1) ** 2).mean() + 
                 (model(torch.cat((torch.full_like(x, domain[1]), t), 1)).squeeze(1) ** 2).mean())  # Dirichlet BCs

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss {loss.item()}')

train_model(model, 5000, [-10, 10], [0, 5])  # Training for 5000 epochs

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
    
    t_tensor = torch.full((len(x_numerical), 1), t, dtype=torch.float32)
    x_tensor = torch.Tensor(x_numerical).unsqueeze(1)
    x_t = torch.cat((x_tensor, t_tensor), 1)
    with torch.no_grad():
        u_dgm = model(x_t).numpy()

    line1.set_ydata(u_numerical)
    line2.set_ydata(u_dgm.flatten())
    return line1, line2,

ani = FuncAnimation(fig, animate, frames=len(time_steps), interval=50, blit=True, repeat=False)
plt.show()
