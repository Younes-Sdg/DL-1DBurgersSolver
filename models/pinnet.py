import torch
import torch.nn as nn
import torch.autograd as autograd

class PINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=50, layers=3):
        super(PINN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()

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
        u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        # Burgers' equation residual
        residual = u_t + u_pred * u_x - (0.01 / torch.pi) * u_xx
        
        # Loss is a combination of the data loss and the PDE residual loss
        loss = nn.MSELoss()(u_pred, u) + torch.mean(residual ** 2)
        return loss
