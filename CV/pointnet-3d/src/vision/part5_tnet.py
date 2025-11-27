from typing import Tuple

import torch
from vision.part3_pointnet import PointNet
from torch import nn


class TNet(nn.Module):

    def __init__(self,
        in_dim: int=3,
        hidden_dims: Tuple[int, int, int]=(64, 128, 1024),
        regression_dims: Tuple[int, int]=(512, 256),
        pts_per_obj=200
    ) -> None:
        '''
        Constructor for TNet to define layers.

        Hint: The architecture is almost the same as your PointNet, just with a different
              output dimension

        Just like with PointNet, you will need to repeat the first hidden dim.
        See mlp(64, 64) in the diagram. Furthermore, you will want to include
        a BatchNorm1d after each layer in the encoder except for the final layer
        for easier training.


        Args:
        -   classes: Number of output classes
        -   in_dim: Input dimensionality for points.
        -   hidden_dims: The dimensions of the encoding MLPs. This is similar to
                         that of PointNet
        -   regression_dims: The dimensions of regression MLPs. This is similar
                         to the classifier dims in PointNet
        -   pts_per_obj: The number of points that each point cloud is padded to
        '''
        super().__init__()

        self.encoder_head = None
        self.regression_head = None
        self.in_dim = None

        ############################################################################
        # Student code begin
        ############################################################################

        self.in_dim = in_dim
        
        encoder_layers = []
        encoder_layers.append(nn.Linear(in_dim, hidden_dims[0]))
        encoder_layers.append(nn.BatchNorm1d(hidden_dims[0]))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(hidden_dims[0], hidden_dims[0]))
        encoder_layers.append(nn.BatchNorm1d(hidden_dims[0]))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(hidden_dims[0], hidden_dims[1]))
        encoder_layers.append(nn.BatchNorm1d(hidden_dims[1]))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(hidden_dims[1], hidden_dims[2]))
        self.encoder_head = nn.Sequential(*encoder_layers)
        
        regression_layers = []
        regression_layers.append(nn.Linear(hidden_dims[2], regression_dims[0]))
        regression_layers.append(nn.ReLU())
        regression_layers.append(nn.Linear(regression_dims[0], regression_dims[1]))
        regression_layers.append(nn.ReLU())
        final_layer = nn.Linear(regression_dims[1], in_dim * in_dim)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        regression_layers.append(final_layer)
        self.regression_head = nn.Sequential(*regression_layers)

        ############################################################################
        # Student code end
        ############################################################################


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the T-Net. Compute the transformation matrices, but do not apply them yet.
        The forward pass is the same as that of your original PointNet, except for:
        1) Adding an identity matrix (be sure to set the device to x.device)
        2) Reshaping the output

        Args:
            x: tensor of shape (B, N, in_dim), where B is the batch size, N is the number of points per
               point cloud, and in_dim is the input point dimension

        Output:
        -   transform_matrices: tensor of shape (B, in_dim, in_dim) containing transformation matrices
                       These will be used to transform the point cloud.
        '''

        transform_matrices = None

        ############################################################################
        # Student code begin
        ############################################################################

        B, N, in_dim = x.shape
        x_flat = x.view(B * N, in_dim)
        encoded_flat = self.encoder_head(x_flat)
        feature_dim = encoded_flat.shape[1]
        encoded = encoded_flat.view(B, N, feature_dim)
        global_feature = torch.max(encoded, dim=1)[0]
        transform_flat = self.regression_head(global_feature)
        transform_matrices = transform_flat.view(B, self.in_dim, self.in_dim)
        identity = torch.eye(self.in_dim, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        transform_matrices = transform_matrices + identity

        ############################################################################
        # Student code end
        ############################################################################

        return transform_matrices


class PointNetTNet(nn.Module):

    def __init__(
        self,
        classes: int,
        in_dim: int=3,
        hidden_dims: Tuple[int, int, int]=(64, 128, 1024),
        classifier_dims: Tuple[int, int]=(512, 256),
        tnet_hidden_dims: Tuple[int, int, int]=(64, 128, 1024),
        tnet_regression_dims: Tuple[int, int]=(512, 256),
        pts_per_obj=200
    ) -> None:
        '''
        Constructor for PointNet with T-Net. The main difference between our
        original PointNet model and this one is the addition of a T-Net to predict
        a transform to apply to the input point cloud.

        Hint:
        1) Think about how to drectly reuse your PointNet implementation from earlier

        Args:
        -   classes: Number of output classes
        -   hidden_dims: The dimensions of the encoding MLPs.
        -   classifier_dims: The dimensions of classifier MLPs.
        -   tnet_hidden_dims: The dimensions of the encoding MLPs for T-Net
        -   tnet_regression_dims: The dimensions of the regression MLPs for T-Net
        -   pts_per_obj: The number of points that each point cloud is padded to
        '''
        super().__init__()

        self.tnet = None
        self.point_net = None

        ############################################################################
        # Student code begin
        ############################################################################

        self.tnet = TNet(in_dim, tnet_hidden_dims, tnet_regression_dims, pts_per_obj)
        self.point_net = PointNet(classes, in_dim, hidden_dims, classifier_dims, pts_per_obj)

        ############################################################################
        # Student code end
        ############################################################################


    def apply_tnet(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the transformation matrices by passing x into T-Net, and
        compute the transformed points by batch matrix multiplying x by the
        transformation matrices.

        Hint: Use torch.bmm for batched matrix multiplication. Multiply x by
        the transformation matrix rather than the other way around.

        Args:
        -   x: tensor of shape (B, pts_per_obj, 3), where B is the batch size and
               pts_per_obj is the number of points per point cloud

        Outputs:
        -   x_transformed: tensor of shape (B, pts_per_obj, 3) containing the
                           transformed point clouds per object.
        '''
        x_transformed = None

        ############################################################################
        # Student code begin
        ############################################################################

        transform_matrices = self.tnet(x)
        x_transformed = torch.bmm(x, transform_matrices)

        ############################################################################
        # Student code end
        ############################################################################

        return x_transformed


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the PointNet model.

        Hint:
        1) Apply the T-Net transforms via apply_tnet
        2) Use your original PointNet architecture on the transformed pointcloud

        Args:
        -   x: tensor of shape (B, pts_per_obj, 3), where B is the batch size and
               pts_per_obj is the number of points per point cloud

        Outputs:
        -   class_outputs: tensor of shape (B, classes) containing raw scores for each class
        -   encodings: tensor of shape (B, N, hidden_dims[-1]), the final vector for each input point
                       before global maximization. This will be used later for analysis.
        '''
        class_outputs = None
        encodings = None

        ############################################################################
        # Student code begin
        ############################################################################

        x_transformed = self.apply_tnet(x)
        class_outputs, encodings = self.point_net(x_transformed)

        ############################################################################
        # Student code end
        ############################################################################

        return class_outputs, encodings

