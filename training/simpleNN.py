import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from tqdm import tqdm

from models.simple_nn import SimpleNN  # Ensure this import points to where your SimpleNN class is located

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training will use: {device}")

# Initialize the SimpleNN model
model = SimpleNN(input_dim=2, hidden_dim=50, layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Load real solution data
data = pd.read_csv('simulation_data.csv')
x_real = torch.tensor(data['x'].values, dtype=torch.float32).unsqueeze(1).to(device)
t_real = torch.tensor(data['time'].values, dtype=torch.float32).unsqueeze(1).to(device)
u_real = torch.tensor(data['u'].values, dtype=torch.float32).unsqueeze(1).to(device)

# Combine x and t into a single input tensor
inputs = torch.cat((x_real, t_real), dim=1)

# Training loop
def train_model(model, epochs, inputs, u_real):
    model.train()
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        optimizer.zero_grad()
        u_pred = model(inputs)
        loss = criterion(u_pred, u_real)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            tqdm.write(f'Epoch {epoch}: Loss {loss.item()}')

epochs = 1500
train_model(model, epochs, inputs, u_real)

# Visualization
x_numerical = data['x'].unique()
time_steps = data['time'].unique()

fig, ax = plt.subplots(figsize=(10, 6))
line1, = ax.plot(x_numerical, np.zeros_like(x_numerical), 'r-', label='Numerical Solution')
line2, = ax.plot(x_numerical, np.zeros_like(x_numerical), 'b--', label='NN Solution')
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
        u_nn = model(x_t).cpu().numpy()

    line1.set_ydata(u_numerical)
    line2.set_ydata(u_nn.flatten())
    return line1, line2,

ani = FuncAnimation(fig, animate, frames=len(time_steps), interval=50, blit=True, repeat=False)
plt.show()
