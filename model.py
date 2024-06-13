import torch.nn as nn
import torch
import torch.nn.functional as F
import encoder
import nerf

class mrirecon(nn.Module):
    def __init__(self, feature_dim, decoder_depth, decoder_width):
        super(mrirecon, self).__init__()
        self.encoder = encoder.RDN(feature_dim=feature_dim)
        positional_encoding_dim = 3 + 2 * 3 * 6
        self.decoder = nerf.NeRFDecoder(input_dim=feature_dim + positional_encoding_dim, 
                                        hidden_dim=decoder_width, output_dim=1, 
                                        num_layers=decoder_depth)


    def forward(self, img_lr, xyz_hr):
            feature_map = self.encoder(img_lr) 
            
            # Generate feature vector for coordinate through trilinear interpolation
            feature_vector = F.grid_sample(feature_map, xyz_hr.flip(-1).unsqueeze(1).unsqueeze(1),
                                        mode='bilinear',
                                        align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)  # N×K×feature_dim
            
            # Apply positional encoding to coordinates
            xyz_hr_encoded = nerf.positional_encoding(xyz_hr)  # N×K×positional_encoding_dim
            
            # Concatenate feature vector with encoded coordinates
            feature_vector_and_xyz_hr = torch.cat([feature_vector, xyz_hr_encoded], dim=-1)  # N×K×(feature_dim + positional_encoding_dim)
            
            # Estimate the voxel intensity at the coordinate using decoder
            N, K = xyz_hr.shape[:2]
            intensity_pre = self.decoder(feature_vector_and_xyz_hr.view(N * K, -1)).view(N, K, -1)
            
            return intensity_pre