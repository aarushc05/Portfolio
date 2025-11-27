from typing import Tuple

import torch
from torch import nn


class PointNet(nn.Module):
    '''
    A simplified version of PointNet (https://arxiv.org/abs/1612.00593)
    Ignoring the transforms and segmentation head.
    '''
    def __init__(self,
        classes: int,
        in_dim: int=3,
        hidden_dims: Tuple[int, int, int]=(64, 128, 1024),
        classifier_dims: Tuple[int, int]=(512, 256),
        pts_per_obj=200
    ) -> None:
        '''
        Constructor for PointNet to define layers.

        Hint: See the modified PointNet architecture diagram from the pdf.
        You will need to repeat the first hidden dim (see mlp(64, 64) in the diagram).
        Furthermore, you will want to include a BatchNorm1d after each layer in the encoder
        except for the final layer for easier training.

        Args:
        -   classes: Number of output classes
        -   in_dim: Input dimensionality for points. This parameter is 3 by default for
                    for the basic PointNet.
        -   hidden_dims: The dimensions of the encoding MLPs.
        -   classifier_dims: The dimensions of classifier MLPs.
        -   pts_per_obj: The number of points that each point cloud is padded to
        '''
        super().__init__()

        self.encoder_head = None
        self.classifier_head = None

        ############################################################################
        # Student code begin
        ############################################################################

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
        
        classifier_layers = []
        classifier_layers.append(nn.Linear(hidden_dims[2], classifier_dims[0]))
        classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Linear(classifier_dims[0], classifier_dims[1]))
        classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Linear(classifier_dims[1], classes))
        self.classifier_head = nn.Sequential(*classifier_layers)

        ############################################################################
        # Student code end
        ############################################################################


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the PointNet model.

        Args:
            x: tensor of shape (B, N, in_dim), where B is the batch size, N is the number of points per
               point cloud, and in_dim is the input point dimension

        Output:
        -   class_outputs: tensor of shape (B, classes) containing raw scores for each class
        -   encodings: tensor of shape (B, N, hidden_dims[-1]), the final vector for each input point
                       before global maximization. This will be used later for analysis.
        '''

        class_outputs = None
        encodings = None

        ############################################################################
        # Student code begin
        ############################################################################

        B, N, in_dim = x.shape
        x_flat = x.view(B * N, in_dim)
        encoded_flat = self.encoder_head(x_flat)
        feature_dim = encoded_flat.shape[1]
        encodings = encoded_flat.view(B, N, feature_dim)
        global_feature = torch.max(encodings, dim=1)[0]
        class_outputs = self.classifier_head(global_feature)

        ############################################################################
        # Student code end
        ############################################################################

        return class_outputs, encodings
