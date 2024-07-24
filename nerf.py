import torch
import torch.nn as nn

def positional_encoding(coords, num_encoding_functions=6, include_input=True):
    """
    Apply positional encoding to coordinates.
    
    Parameters
    ----------
    coords : torch.Tensor
        Coordinates tensor.
    num_encoding_functions : int, optional
        Number of encoding functions to apply (default is 6).
    include_input : bool, optional
        Whether to include the input coordinates in the encoding (default is True).
    
    Returns
    -------
    torch.Tensor
        Positional encoded coordinates.
    """
    encoding = [coords] if include_input else []
    for i in range(num_encoding_functions):
        for func in [torch.sin, torch.cos]:
            encoding.append(func(2.0 ** i * coords))
    return torch.cat(encoding, dim=-1)

class NeRFDecoder(nn.Module):
    """
    NeRF-inspired Decoder Network.
    
    Parameters
    ----------
    input_dim : int
        Dimension of the input feature vector.
    hidden_dim : int
        Dimension of the hidden layers.
    output_dim : int
        Dimension of the output (e.g., intensity).
    num_layers : int
        Number of layers in the decoder network.
    """
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
        """
        Forward pass through the NeRFDecoder.
        
        Returns
        -------
        torch.Tensor
            Output tensor after passing through the decoder network.
        """
        return self.model(x)
