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

from clay.model import Encoder


class JEPASegmentEncoder(nn.Module):
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
            nn.ConvTranspose2d(self.vit_encoder.student_encoder.dim, self.vit_encoder.student_encoder.dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.vit_encoder.student_encoder.dim),
            nn.GELU(),
            nn.ConvTranspose2d(self.vit_encoder.student_encoder.dim, self.vit_encoder.student_encoder.dim, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.vit_encoder.student_encoder.dim, self.vit_encoder.student_encoder.dim, kernel_size=2, stride=2),
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

       
        x = self.vit_encoder.init_norm(cube)
        #get the patch embeddings
        x = self.vit_encoder.patch_embed(x)
        b, n, e = x.shape
        #add the positional embeddings
        x = x + self.vit_encoder.pos_embedding
        #normalize the embeddings
        x = self.vit_encoder.post_emb_norm(x)

        features = []
        _, _cube = self.vit_encoder.student_encoder.forward(x, return_hiddens=True)
        _cube = _cube.layer_hiddens 

        print(len(_cube)) 
        for idx in range(len(_cube)):
            if idx in self.feature_maps:
                tmp_cube = rearrange(
                    _cube[idx], "B (H W) D -> B D H W", H=H // 3, W=W // 3
                )
                features.append(tmp_cube)
        # Apply FPN layers
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4, self.fpn5]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        return features


class JEPASegmentor(nn.Module):
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
        self.encoder = JEPASegmentEncoder(
            vit_encoder,
            feature_maps=feature_maps,
        )
        self.upsamples = [nn.Upsample(scale_factor=2**i) for i in range(3)] #+ [
            #nn.Upsample(scale_factor=4),
        #]

        chan_mult = len(feature_maps) # + 1

        self.fusion = nn.Conv2d(self.encoder.vit_encoder.student_encoder.dim *chan_mult, self.encoder.vit_encoder.student_encoder.dim, kernel_size=1)
        self.seg_head = nn.Conv2d(self.encoder.vit_encoder.student_encoder.dim, num_classes, kernel_size=1)

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
