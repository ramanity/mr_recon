import torch
import torch.nn as nn
import torch.nn.functional as F
import encoder
import nerf

class MRIRecon(nn.Module):
    """
    MRI Reconstruction Neural Network Model.

    This model uses a Residual Dense Network (RDN) as the encoder
    and a NeRF-inspired decoder to reconstruct MRI images from
    low-resolution inputs and high-resolution coordinates.

    Parameters
    ----------
    feature_dim : int
        Dimension of the feature map output by the encoder.
    decoder_depth : int
        Number of layers in the decoder network.
    decoder_width : int
        Width of the hidden layers in the decoder network.
    """
    def __init__(self, feature_dim, decoder_depth, decoder_width):
        super(MRIRecon, self).__init__()
        self.encoder = encoder.RDN(feature_dim=feature_dim)
        positional_encoding_dim = 3 + 2 * 3 * 6
        self.decoder = nerf.NeRFDecoder(input_dim=feature_dim + positional_encoding_dim, 
                                        hidden_dim=decoder_width, output_dim=1, 
                                        num_layers=decoder_depth)

    def forward(self, img_lr, xyz_hr):
        """
        Forward pass through the MRI reconstruction network.

        Parameters
        ----------
        img_lr : torch.Tensor
            Low-resolution input image tensor.
        xyz_hr : torch.Tensor
            High-resolution coordinates tensor.

        Returns
        -------
        torch.Tensor
            Reconstructed voxel intensity predictions.
        """
        # Encode low-resolution image to feature map
        feature_map = self.encoder(img_lr)
        
        # Generate feature vector for coordinate through trilinear interpolation
        feature_vector = F.grid_sample(feature_map, xyz_hr.flip(-1).unsqueeze(1).unsqueeze(1),
                                       mode='bilinear', align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
        
        # Apply positional encoding to coordinates
        xyz_hr_encoded = nerf.positional_encoding(xyz_hr)
        
        # Concatenate feature vector with encoded coordinates
        feature_vector_and_xyz_hr = torch.cat([feature_vector, xyz_hr_encoded], dim=-1)
        
        # Estimate the voxel intensity at the coordinate using the decoder
        N, K = xyz_hr.shape[:2]
        intensity_pre = self.decoder(feature_vector_and_xyz_hr.view(N * K, -1)).view(N, K, -1)
        
        return intensity_pre
