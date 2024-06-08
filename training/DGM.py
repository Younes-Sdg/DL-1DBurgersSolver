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
import time

# Setup device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training will use: {device}")

# Initialize the model and set up for tracking the best model
model = DGMNet(input_dim=2, hidden_dim=50, layers=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

min_loss = float('inf')
best_model_state = None

# Record start time
start_time = time.time()

def train_model(model, epochs, domain, time_frame, num_points):
    global min_loss, best_model_state
    
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

        if loss.item() < min_loss:
            min_loss = loss.item()
            best_model_state = model.state_dict()

# Training parameters
epochs = 15000
space_domain = [-10, 10]
time_domain = [0, 5]
num_points = 1000

# Start training
train_model(model, epochs, space_domain, time_domain, num_points)

# Calculate training time in minutes
end_time = time.time()
training_time_minutes = (end_time - start_time) / 60

# Save the best model
if not os.path.exists('trained_models'):
    os.makedirs('trained_models')
torch.save(best_model_state, 'trained_models/dgm_best.pth')

# Create a text file to save hyperparameters and the best loss
params_path = 'trained_models/dgm_best_parameters.txt'
with open(params_path, 'w') as f:
    f.write(f"Training Time (minutes): {training_time_minutes:.2f}\n")
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Learning Rate: {0.001}\n")
    f.write(f"Weight Decay: {1e-4}\n")
    f.write(f"Step Size: {1000}\n")
    f.write(f"Gamma: {0.5}\n")
    f.write(f"Input Dimension: {2}\n")
    f.write(f"Hidden Dimension: {50}\n")
    f.write(f"Number of Layers: {8}\n")
    f.write(f"Best Loss: {min_loss:.6f}\n")




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
ax.set_title(f"Burgers' Equation Solutions\nTraining Time: {training_time_minutes:.2f} minutes")
ax.set_xlabel('X')
ax.set_ylabel('U')
ax.legend()

model.load_state_dict(torch.load('trained_models/dgm_best.pth'))

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
output_path = os.path.join('animations', 'dgm_animation.gif')
ani.save(output_path, writer='pillow', fps=10)
print("Animation saved to:", output_path)

plt.show()
