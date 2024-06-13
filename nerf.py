import torch
import torch.nn as nn


def positional_encoding(coords, num_encoding_functions=6, include_input=True):
    encoding = [coords] if include_input else []
    for i in range(num_encoding_functions):
        for func in [torch.sin, torch.cos]:
            encoding.append(func(2.0 ** i * coords))
    return torch.cat(encoding, dim=-1)

class NeRFDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(NeRFDecoder, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)