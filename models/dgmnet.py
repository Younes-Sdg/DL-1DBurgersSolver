import torch
import torch.nn as nn
import torch.autograd as autograd

class DGMNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=100, layers=4):
        super(DGMNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        S = self.activation(self.input_layer(x))
        for hidden in self.hidden_layers:
            Z = self.activation(hidden(S))
            G = self.activation(hidden(S) + hidden(Z))
            R = self.activation(hidden(S) + hidden(G))
            H = self.activation(hidden(S) + hidden(R))
            S = (1 - G) * H + Z
        output = self.output_layer(S)
        return output

    def loss(self, x, t, ic):
        x_t = torch.cat([x, t], dim=1)
        u_pred = self.forward(x_t)
        
        # Compute gradients for enforcing PDE constraints
        grad_outputs = torch.ones_like(u_pred, requires_grad=False)
        u_t = autograd.grad(u_pred, t, grad_outputs=grad_outputs, create_graph=True)[0]
        u_x = autograd.grad(u_pred, x, grad_outputs=grad_outputs, create_graph=True)[0]

        # Burgers' equation residual without viscosity
        residual = u_t + (u_pred * u_x)
        loss_pde = torch.mean(residual ** 2)

        # Initial condition loss at t=0
        t_zero_mask = (t == 0).squeeze()
        loss_ic = nn.MSELoss()(u_pred[t_zero_mask], ic[t_zero_mask])

        # Boundary conditions handled directly
        x_left = torch.full_like(x, -10)  # left boundary at x = -10
        x_right = torch.full_like(x, 10)  # right boundary at x = 10
        t_bc = t  # using the same timesteps for boundary conditions
        x_left_t = torch.cat((x_left, t_bc), dim=1)
        x_right_t = torch.cat((x_right, t_bc), dim=1)
        
        u_bc_left = self.forward(x_left_t)
        u_bc_right = self.forward(x_right_t)
        loss_bc_left = nn.MSELoss()(u_bc_left, torch.full_like(u_bc_left, 0.2))
        loss_bc_right = nn.MSELoss()(u_bc_right, torch.full_like(u_bc_right, 0.4))
        
        # Total loss
        loss = loss_pde + loss_ic + loss_bc_left + loss_bc_right
        return loss
