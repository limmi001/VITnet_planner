"""
OL Network - Based on LSTMNetVIT
Input: depth image
Output: endstate with shape [batch, 3, 20]
"""

import torch
from torch import nn
from .models.backbone import LSTMNetVIT


class OLNetwork(nn.Module):
    """
    OL Network using LSTMNetVIT backbone
    Input: depth image (batch, 1, H, W)
    Output: endstate (batch, 3, 20)
    """

    def __init__(self):
        super(OLNetwork, self).__init__()
        # Use LSTMNetVIT as backbone
        # LSTMNetVIT already outputs (batch, 3, 20) directly
        self.backbone = LSTMNetVIT()

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of OL Network
        Args:
            depth: input depth image, shape (batch, 1, H, W)
        
        Returns:
            endstate: output with shape (batch, 3, 20)
        """
        # Forward through backbone (LSTMNetVIT already outputs (batch, 3, 20))
        endstate, h = self.backbone(depth)
        
        return endstate


if __name__ == '__main__':
    # Test the network
    net = OLNetwork()
    
    # Test with input depth (batch=2, 1, 60, 90) - standard size from LSTMNetVIT
    depth = torch.randn(2, 1, 60, 90)
    
    endstate = net(depth)
    print(f"Input depth shape: {depth.shape}")
    print(f"Output endstate shape: {endstate.shape}")
    print(f"Expected endstate shape: torch.Size([2, 3, 20])")
    print(f"Shape match: {endstate.shape == torch.Size([2, 3, 20])}")
