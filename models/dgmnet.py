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

    def loss(self, x_t, ic, boundary_conditions):
        u_pred = self.forward(x_t)
        grads = autograd.grad(outputs=u_pred, inputs=x_t,
                              grad_outputs=torch.ones_like(u_pred),
                              create_graph=True, only_inputs=True)
        u_t, u_x = grads[0][:, 1], grads[0][:, 0]

        # Residual of the PDE
        residual = u_t + u_pred * u_x
        loss_pde = torch.mean(residual ** 2)

        # Initial condition loss
        loss_ic = torch.mean((u_pred - ic) ** 2)

        # Boundary conditions
        bc_left = boundary_conditions[0]
        bc_right = boundary_conditions[1]
        u_bc_left = self.forward(bc_left)
        u_bc_right = self.forward(bc_right)
        loss_bc = torch.mean((u_bc_left - 0.2) ** 2) + torch.mean((u_bc_right - 0.4) ** 2)

        return loss_pde + loss_ic + loss_bc
