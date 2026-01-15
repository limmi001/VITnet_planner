"""
OL Network - Based on LSTMNetVIT
Input: depth image
Output: endstate with shape [batch, 3, 20]
"""

import torch
from torch import nn
from typing import Optional
from .models.backbone import LSTMNetVIT
from policy.ol_state_transform import OLStateTransform


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
        self.state_backbone = nn.Sequential()
        self.backbone = LSTMNetVIT()

    def forward(self, depth: torch.Tensor, obs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of OL Network
        Args:
            depth: input depth image, shape (batch, 1, H, W)
            obs: optional observation tensor, shape (batch, obs_dim)
        
        Returns:
            endstate: output with shape (batch, 3, 20)
        """
        # Process obs through state_backbone if provided
        obs_feature = None
        if obs is not None:
            obs_feature = self.state_backbone(obs)
        
        # Forward through backbone with optional obs_feature
        endstate, h = self.backbone(depth, obs_feature)
        
        return endstate

    def inference(self, depth: torch.Tensor, obs: Optional[torch.Tensor] = None, yaw_baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inference helper: run forward and map predicted spherical deltas
        to Cartesian positions using `OLStateTransform`.

        Args:
            depth: input depth image, shape (B,1,H,W)
            obs: optional observation tensor, shape (B, obs_dim)
            yaw_baseline: optional tensor of shape [B] specifying base yaw (radians)

        Returns:
            positions: tensor of shape [B,3,N]
        """
        endstate = self.forward(depth, obs)  # [B,3,N]
        transformer = OLStateTransform()
        positions = transformer.pred_to_cartesian(endstate, yaw_baseline)
        return positions


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
