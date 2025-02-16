"""
Clay Segmentor for semantic segmentation tasks.

Attribution:
Decoder from Segformer: Simple and Efficient Design for Semantic Segmentation
with Transformers
Paper URL: https://arxiv.org/abs/2105.15203
"""

import re
import torch.nn.functional as F
import torch
from einops import rearrange, repeat
from torch import nn



class CDBNSegmentEncoder(nn.Module):
    """
    Encoder class for segmentation tasks, incorporating a feature pyramid
    network (FPN).

    Attributes:
        feature_maps (list): Indices of layers to be used for generating
        feature maps.
    """

    def __init__(  # noqa: PLR0913
        self,
        cdbn_encoder, 
        feature_maps,
    ):
        super().__init__()

        self.feature_maps = feature_maps
        self.cdbn_encoder = cdbn_encoder
   

        # Define Feature Pyramid Network (FPN) layers
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(100, 300, kernel_size=2, stride=2),
            nn.BatchNorm2d(300),
            nn.GELU(),
            nn.ConvTranspose2d(300, 200, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(200, 200, kernel_size=2, stride=2),
        )

        self.fpn3 = nn.Identity()

        self.fpn4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fpn5 = nn.Identity()

        # Set device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def forward(self, cube):
        """
        Forward pass of the SegmentEncoder.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            list: A list of feature maps extracted from the datacube.
        """
        tmp_cube = torch.squeeze(cube)
        mx_tile = 0
        features = []
        tile_d1 = 1
        tile_d2 = 2
        if tmp_cube.ndim > 3:
            tile_d1 = 2
            tile_d2 = 3
        for i in range(len(self.cdbn_encoder.models)):
            tmp_cube = self.cdbn_encoder.models[i](tmp_cube)
            if i in self.feature_maps:
                features.append(tmp_cube)
                if tmp_cube.ndim == 3:
                    tmp_cube = torch.unsqueeze(tmp_cube,0)
                print("HERE TILING ISSUE", tmp_cube.shape)
                mx_tile = max(tmp_cube.shape[tile_d2], mx_tile)

        for i in range(len(self.cdbn_encoder.models)):
            if features[i].shape[tile_d1] != mx_tile or features[i].shape[tile_d2] != mx_tile:
                features[i] = F.interpolate(
                    features[i],
                    size=(mx_tile, mx_tile),
                    mode="bilinear",
                    align_corners=False,
                )  # Resize to match labels size
 
        # Apply FPN layers
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4, self.fpn5]
        for i in range(len(features)):
            features[i] = ops[i](features[i])
            print("FINAL ENCODER FEATURES", features[i].shape)

        return features


class CDBNSegmentor(nn.Module):
    """
    Clay Segmentor class that combines the Encoder with FPN layers for semantic
    segmentation.

    Attributes:
        num_classes (int): Number of output classes for segmentation.
        feature_maps (list): Indices of layers to be used for generating feature maps.
    """

    def __init__(self, num_classes, feature_maps, cdbn_encoder):
        super().__init__()
        # Default values are for the clay mae base model.
        self.encoder = CDBNSegmentEncoder(
            cdbn_encoder,
            feature_maps=feature_maps,
        )
        self.upsamples = [nn.Upsample(scale_factor=2**i) for i in range(3)] #+ [
            #nn.Upsample(scale_factor=4),
        #]

        chan_mult = len(feature_maps)  + 6

        self.fusion = nn.Conv2d(400,900, kernel_size=1)
        self.seg_head = nn.Conv2d(900, num_classes, kernel_size=1)

    def forward(self, datacube):
        """
        Forward pass of the Segmentor.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The segmentation logits.
        """
        features = self.encoder(datacube)
        for i in range(len(features)):
            features[i] = self.upsamples[i](features[i])
            print(features[i].shape)

        fused = torch.cat(features, dim=1)
        fused = self.fusion(fused)

        logits = self.seg_head(fused)
        return logits
