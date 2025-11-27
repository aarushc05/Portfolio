from typing import Tuple

import torch
from torch import nn


class Baseline(nn.Module):
    '''
    A simple baseline that counts points per voxel in the point cloud
    and then uses a linear classifier to make a prediction
    '''
    def __init__(self,
        classes: int,
        voxel_resolution=4,
        mode="count"
    ) -> None:
        '''
        Constructor for Baseline to define layers.

        Args:
        -   classes: Number of output classes
        -   voxel_resolution: Number of positions per dimension to count
        -   mode: Whether to count the number of points per voxel ("count") or just check binary occupancy ("occupancy")
        '''
        assert mode in ["count", "occupancy"]

        super().__init__()

        self.classifier = None
        self.voxel_resolution = None
        self.mode = None

        ############################################################################
        # Student code begin
        ############################################################################

        self.voxel_resolution = voxel_resolution
        self.mode = mode
        self.classifier = nn.Linear(voxel_resolution ** 3, classes)

        ############################################################################
        # Student code end
        ############################################################################


    def count_points(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Create the feature as input to the linear classifier by counting the number of points per voxel.
        This is effectively taking a 3D histogram for every item in a batch.

        Hint: 
        1) torch.histogramdd will be useful here

        Args:
        -   x: tensor of shape (B, N, in_dim)

        Output:
        -   counts: tensor of shape (B, voxel_resolution**3), indicating the percentage of points that landed in each voxel
        '''

        counts = None

        ############################################################################
        # Student code begin
        ############################################################################

        B, N, _ = x.shape
        counts_list = []
        
        for b in range(B):
            sample_points = x[b]
            x_coords = sample_points[:, 0]
            y_coords = sample_points[:, 1]
            z_coords = sample_points[:, 2]
            
            x_bins = torch.clamp(((x_coords + 2.0) / 4.0) * self.voxel_resolution, 0, self.voxel_resolution - 1).long()
            y_bins = torch.clamp(((y_coords + 2.0) / 4.0) * self.voxel_resolution, 0, self.voxel_resolution - 1).long()
            z_bins = torch.clamp(((z_coords + 2.0) / 4.0) * self.voxel_resolution, 0, self.voxel_resolution - 1).long()
            
            hist = torch.zeros((self.voxel_resolution, self.voxel_resolution, self.voxel_resolution), device=x.device)
            for i in range(N):
                hist[x_bins[i], y_bins[i], z_bins[i]] += 1
            
            hist_flat = hist.flatten()
            hist_normalized = hist_flat / N
            counts_list.append(hist_normalized)
        
        counts = torch.stack(counts_list)

        ############################################################################
        # Student code end
        ############################################################################

        return counts


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the Baseline model. Make sure you handle the case where the mode 
        is set to "occupancy" by thresholding the result of count_points on zero.

        Args:
            x: tensor of shape (B, N, 3), where B is the batch size and N is the number of points per
               point cloud
        Output:
        -   class_outputs: tensor of shape (B, classes) containing raw scores for each class
        -   None, just used to allow reuse of training infrastructure
        '''

        class_outputs = None

        ############################################################################
        # Student code begin
        ############################################################################

        counts = self.count_points(x)
        if self.mode == "occupancy":
            counts = (counts > 0).float()
        class_outputs = self.classifier(counts)

        ############################################################################
        # Student code end
        ############################################################################

        return class_outputs, None
