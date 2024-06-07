import torch
import torch.nn as nn
import torch.autograd as autograd
from tqdm import tqdm
from data_generator import functions
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=50, layers=3):
        super(PINN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU ()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for hidden in self.hidden_layers:
            x = self.activation(hidden(x))
        output = self.output_layer(x)
        return output

    def loss(self, x, t, u):
        u_pred = self.forward(torch.cat([x, t], dim=1))
        
        # Compute gradients for enforcing PDE constraints
        u_t = autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_x = autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        
        # Burgers' equation residual without viscosity term
        residual = u_t + u_pred * u_x
        t_zero_mask = (t == 0).squeeze()
        u_pred_t0 = u_pred[t_zero_mask]
        u_t0 = u[t_zero_mask]
        # Loss is a combination of the data loss and the PDE residual loss
        loss = nn.MSELoss()(u_pred_t0, u_t0) + torch.mean(residual ** 2)
        return loss



def train_model(model, epochs, domain, time_frame, save_path='best_model.pth'):
    """
    Inner function to train the model with detailed loss tracking and model saving.

    Args:
    model (nn.Module): The neural network model to train.
    epochs (int): Number of training epochs.
    domain (list): Spatial domain as [min, max].
    time_frame (list): Time frame as [start, end].
    save_path (str): Path to save the best model.

    Returns:
    list: A list containing the loss from each training epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    model.train()
    loss_history = []
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

        current_loss = loss.item()
        loss_history.append(current_loss)
        
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch}: New best loss {best_loss}")

    return best_loss

def setup_animation(data, x_numerical, time_steps):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Supposons que data contienne déjà les données numériques sous une forme adaptée
    # par exemple, u_numerical pourrait être précalculé pour le premier pas de temps (t = time_steps[0])
    initial_u = data[data['time'] == time_steps[0]]['u'].values
    line1, = ax.plot(x_numerical, initial_u, 'r-', label='Numerical Solution')
    # Pour la solution PINN, supposons une initialisation à zéro ou une autre initialisation logique
    line2, = ax.plot(x_numerical, initial_u, 'b--', label='PINN Solution')  # Modifier ceci selon votre modèle initial
    ax.set_xlim(min(x_numerical), max(x_numerical))
    ax.set_ylim(min(data['u']) - 0.1, max(data['u']) + 0.1)
    ax.set_title("Burgers' Equation Solutions")
    ax.set_xlabel('X')
    ax.set_ylabel('U')
    ax.legend()
    return fig, ax, line1, line2

def pinns_mse_to_numerical_solution(hid_dim, lay,epochs, fichier):
    """
    Function to train a Physics Informed Neural Network (PINN) with specified dimensions and layers,
    and compute the maximum Mean Squared Error (MSE) to the numerical solution from a given dataset.

    Args:
    hid_dim (int): Number of neurons in each hidden layer.
    lay (int): Number of hidden layers in the model.
    fichier (str): Path to the CSV file containing the numerical solution data.

    Returns:
    float: The maximum Mean Squared Error (MSE) between the PINN predictions and the numerical solution.
    """
    # Setup the device for training (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training will use: {device}")

    # Initialize the PINN model
    model = PINN(input_dim=2, hidden_dim=hid_dim, layers=lay).to(device)

    # Parameters for training
    
    space_domain = [-10, 10]
    time_domain = [0, 5]

    # Training the model
    loss = train_model(model, epochs, space_domain, time_domain)
    print(f'Final training loss = {loss}')

    # Load data
    data = pd.read_csv(fichier)
    x_numerical = data['x'].unique()
    time_steps = data['time'].unique()

    mse_list = []

    for t in time_steps:
        u_numerical = data[data['time'] == t]['u'].values

        t_tensor = torch.full((len(x_numerical), 1), t, dtype=torch.float32).to(device)
        x_tensor = torch.Tensor(x_numerical).unsqueeze(1).to(device)
        x_t_u0 = torch.cat((x_tensor, t_tensor), 1).to(device)
        
        with torch.no_grad():
            u_pinn = model(x_t_u0).cpu().numpy()

        mse = np.mean((u_numerical - u_pinn.flatten())**2)
        mse_list.append(mse)

    # Calculate the maximum MSE
    mse_max = max(mse_list) if mse_list else None
    return mse_max,loss