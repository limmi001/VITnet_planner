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
        self.backbone = LSTMNetVIT()
        
        # Output projection layer
        # LSTMNetVIT outputs (out, h) where out shape is (1, 3) from nn_fc2
        # We need to project this to (batch, 3, 20)
        self.output_projection = nn.Linear(3, 3 * 20)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of OL Network
        Args:
            depth: input depth image, shape (batch, 1, H, W)
        
        Returns:
            endstate: output with shape (batch, 3, 20)
        """
        batch_size = depth.shape[0]
        
        # Forward through backbone (now only needs depth)
        out, h = self.backbone(depth)
        # out shape: (batch, 3) from nn_fc2 Linear layer
        
        # Project to (batch, 3*20) then reshape to (batch, 3, 20)
        endstate = self.output_projection(out)  # (batch, 60)
        endstate = endstate.view(batch_size, 3, 20)  # (batch, 3, 20)
        
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
