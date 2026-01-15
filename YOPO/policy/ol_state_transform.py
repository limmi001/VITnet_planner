import torch
from typing import Optional

from policy.ol_primitive import AnnulusYawPrimitive


class OLStateTransform:
    """
    Transform OL-network predictions (spherical-like deltas) to Cartesian coordinates

    Expected input:
      - endstate_pred: tensor of shape [B, 3, N]
        channel 0: d_r    (radius offset, meters)
        channel 1: d_theta(pitch / zenith angle, radians)
        channel 2: d_phi  (azimuth offset, radians)

    Mapping logic:
      - Each channel vector of length N corresponds to the N rings in
        `AnnulusYawPrimitive` (ring_id 0..N-1). We take the ring anchor
        radius (ring_center) as the base radius and apply d_r additively.
      - A yaw baseline (per-batch) must be provided; the final azimuth
        for each ring is `yaw_baseline + d_phi`.
      - Convert (r, theta, phi) -> Cartesian using
            x = r * cos(theta) * cos(phi_total)
            y = r * cos(theta) * sin(phi_total)
            z = r * sin(theta)

    Returns:
      - positions: tensor of shape [B, 3, N] (x,y,z per ring)
    """

    def __init__(self):
        self.primitive = AnnulusYawPrimitive.get_instance()
        # ring_center shape: [N]
        self.ring_center = self.primitive.ring_center
        self.ring_num = int(self.primitive.ring_num)

    def pred_to_cartesian(self, endstate_pred: torch.Tensor, yaw_baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convert OL-network spherical predictions to Cartesian.

        Args:
            endstate_pred: [B, 3, N] (d_r, d_theta, d_phi)
            yaw_baseline: optional tensor [B] of base yaw for each batch (radians).
                          If None, zeros are used.

        Returns:
            positions: [B, 3, N] (x,y,z)
        """
        assert endstate_pred.dim() == 3 and endstate_pred.size(1) == 3, "endstate_pred must be [B,3,N]"
        B, C, N = endstate_pred.shape
        assert N == self.ring_num, f"expected N={self.ring_num}, got {N}"

        device = endstate_pred.device
        ring_center = self.ring_center.to(device)  # [N]

        d_r = endstate_pred[:, 0, :]      # [B, N]
        d_theta = endstate_pred[:, 1, :]  # [B, N]
        d_phi = endstate_pred[:, 2, :]    # [B, N]

        base_r = ring_center.unsqueeze(0).expand(B, -1)  # [B, N]
        r = base_r + d_r

        if yaw_baseline is None:
            yaw = torch.zeros(B, device=device)
        else:
            yaw = yaw_baseline.to(device).view(B)

        yaw = yaw.unsqueeze(1)  # [B,1]
        phi_total = yaw + d_phi  # [B, N]

        cos_theta = torch.cos(d_theta)
        sin_theta = torch.sin(d_theta)

        x = r * cos_theta * torch.cos(phi_total)
        y = r * cos_theta * torch.sin(phi_total)
        z = r * sin_theta

        pos = torch.stack([x, y, z], dim=1)  # [B, 3, N]
        return pos


__all__ = ["OLStateTransform"]
