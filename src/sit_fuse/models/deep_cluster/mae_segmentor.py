"""
Clay Segmentor for semantic segmentation tasks.

Attribution:
Decoder from Segformer: Simple and Efficient Design for Semantic Segmentation
with Transformers
Paper URL: https://arxiv.org/abs/2105.15203
"""

import re

import torch
from einops import rearrange, repeat
from torch import nn


class MAESegmentEncoder(nn.Module):
    """
    Encoder class for segmentation tasks, incorporating a feature pyramid
    network (FPN).

    Attributes:
        feature_maps (list): Indices of layers to be used for generating
        feature maps.
    """

    def __init__(  # noqa: PLR0913
        self,
        vit_encoder, 
        feature_maps,
    ):
        super().__init__()

        self.feature_maps = feature_maps
        self.vit_encoder = vit_encoder
   

        # Define Feature Pyramid Network (FPN) layers
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.vit_encoder.cls_token.shape[2], self.vit_encoder.cls_token.shape[2], kernel_size=2, stride=2),
            nn.BatchNorm2d(self.vit_encoder.cls_token.shape[2]),
            nn.GELU(),
            nn.ConvTranspose2d(self.vit_encoder.cls_token.shape[2], self.vit_encoder.cls_token.shape[2], kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.vit_encoder.cls_token.shape[2], self.vit_encoder.cls_token.shape[2], kernel_size=2, stride=2),
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
        cube = torch.squeeze(cube)
        B, C, H, W = cube.shape

      
        print(cube.shape)
 
        x = self.vit_encoder.to_patch_embedding(cube)
        b, n, _ = x.shape

        x += self.vit_encoder.pos_embedding[:, :n] #(n + 1)]

        features = []
        idx = 0
        for attn, ff in self.vit_encoder.transformer.layers:
            x = attn(x) + x
            x = ff(x) + x
            if idx in self.feature_maps:
                _cube = x.clone()
                tmp_cube = rearrange(
                    _cube, "B (H W) D -> B D H W", H=H // 1, W=W // 1
                )
                features.append(tmp_cube)
            idx = idx + 1
            print("HERE idx", idx, self.feature_maps, len(features))

        # Apply FPN layers
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4, self.fpn5]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        return features


class MAESegmentor(nn.Module):
    """
    Clay Segmentor class that combines the Encoder with FPN layers for semantic
    segmentation.

    Attributes:
        num_classes (int): Number of output classes for segmentation.
        feature_maps (list): Indices of layers to be used for generating feature maps.
    """

    def __init__(self, num_classes, feature_maps, vit_encoder):
        super().__init__()
        # Default values are for the clay mae base model.
        self.encoder = MAESegmentEncoder(
            vit_encoder,
            feature_maps=feature_maps,
        )
        self.upsamples = [nn.Upsample(scale_factor=2**i) for i in range(len(feature_maps))] #+ [
            #nn.Upsample(scale_factor=4),
        #]

        chan_mult = len(feature_maps) # + 1

        self.fusion = nn.Conv2d(self.encoder.vit_encoder.cls_token.shape[2] *chan_mult, self.encoder.vit_encoder.cls_token.shape[2], kernel_size=1)
        self.seg_head = nn.Conv2d(self.encoder.vit_encoder.cls_token.shape[2], num_classes, kernel_size=1)

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

        fused = torch.cat(features, dim=1)
        fused = self.fusion(fused)

        logits = self.seg_head(fused)
        return logits
