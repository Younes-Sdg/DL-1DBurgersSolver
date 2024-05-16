import torch.nn as nn

class DGMNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=50, layers=3):
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