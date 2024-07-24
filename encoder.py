import torch.nn as nn
import torch

class DenseLayer(nn.Module):
    """
    Dense layer for the Residual Dense Network (RDN).
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)

class RDB(nn.Module):
    """
    Residual Dense Block (RDB) for the RDN.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    growth_rate : int
        Growth rate for the dense layers.
    num_layers : int
        Number of dense layers in the block.
    """
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)]
        )
        self.lff = nn.Conv3d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))

class RDN(nn.Module):
    """
    Implement a Residual Dense Network (RDN) encoder.
    
    Source: "Residual Dense Network for Image Super-Resolution." (Zhang, Yulun, et al.)
    
    Parameters
    ----------
    feature_dim : int, optional
        Dimension of the output features (default is 128).
    num_features : int, optional
        Number of features for shallow feature extraction (default is 64).
    growth_rate : int, optional
        Growth rate for the RDBs (default is 64).
    num_blocks : int, optional
        Number of RDB blocks (default is 8).
    num_layers : int, optional
        Number of dense layers per RDB block (default is 3).
    """
    def __init__(self, feature_dim=128, num_features=64, growth_rate=64, num_blocks=8, num_layers=3):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        self.sfe1 = nn.Conv3d(1, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv3d(num_features, num_features, kernel_size=3, padding=3 // 2)
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))
        self.gff = nn.Sequential(
            nn.Conv3d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv3d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        self.output = nn.Conv3d(self.G0, feature_dim, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)
        x = self.gff(torch.cat(local_features, 1)) + sfe1
        x = self.output(x)
        return x
