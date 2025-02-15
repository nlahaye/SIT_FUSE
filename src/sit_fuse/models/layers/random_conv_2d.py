import torch
import torch.nn as nn
import random

class RandomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        with torch.no_grad():
          self.conv.weight.data.copy_(torch.randn_like(self.conv.weight.data))
          if self.conv.bias is not None:
            self.conv.bias.data.copy_(torch.randn_like(self.conv.bias.data))
        return self.conv(x)

if __name__ == '__main__':
    # Example usage
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    input_size = (1, in_channels, 32, 32)  # Example input size (N, C, H, W)

    # Create an instance of the RandomConv2d layer
    random_conv = RandomConv2d(in_channels, out_channels, kernel_size, padding=1)

    # Generate a random input tensor
    input_tensor = torch.randn(input_size)

    # Perform the random convolution
    output_tensor = random_conv(input_tensor)

    # Print the shape of the output tensor
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
