import torch
import torch.nn as nn

"""
B-spline smoothness loss using second-order and third-order finite difference
- Smoothness (jerk): third-order difference → P_{i+3} - 3P_{i+2} + 3P_{i+1} - P_i
- Acceleration: second-order difference → P_{i+2} - 2P_{i+1} + P_i

For B-spline trajectory optimization in drone navigation
"""

class BSplineSmoothnessLoss(nn.Module):
    def __init__(self, smooth_weight=1.0, accel_weight=1.0):
        """
        Args:
            smooth_weight: 平滑性(jerk)权重，默认1.0
            accel_weight: 加速度权重，默认1.0
        """
        super(BSplineSmoothnessLoss, self).__init__()
        
        # 权重矩阵 (3x3 对角矩阵，xyz三个维度)
        self.register_buffer('R_smooth', torch.eye(3) * smooth_weight)
        self.register_buffer('R_accel', torch.eye(3) * accel_weight)
    
    def forward(self, ctrl_pts):
        """
        计算B样条轨迹的平滑性代价和加速度代价
        
        Args:
            ctrl_pts: (B, 3, N) - B样条控制点
                     B: batch size
                     3: xyz坐标
                     N: 控制点数量
        
        Returns:
            smoothness_cost: (B,) - 平滑性代价（三阶差分，衡量jerk）
            acceleration_cost: (B,) - 加速度代价（二阶差分）
        """
        assert ctrl_pts.dim() == 3, "ctrl_pts must be [B, 3, N]"
        B, dim, N = ctrl_pts.shape
        assert dim == 3, "Expected 3D control points (xyz)"
        
        # ========== 加速度代价 (二阶差分) ==========
        if N >= 3:
            # d2 = P_{i+2} - 2*P_{i+1} + P_i
            d2 = (
                ctrl_pts[:, :, 2:]           # P_{i+2}
                - 2.0 * ctrl_pts[:, :, 1:-1]  # -2*P_{i+1}
                + ctrl_pts[:, :, :-2]         # P_i
            )  # (B, 3, N-2)
            
            # 计算加权平方和: d2^T * R_accel * d2
            accel_cost = torch.einsum("bcn,cd,bdn->bn", d2, self.R_accel, d2)  # (B, N-2)
            acceleration_cost = accel_cost.mean(dim=1)  # (B,)
        else:
            acceleration_cost = torch.zeros(B, device=ctrl_pts.device)
        
        # ========== 平滑性代价 (三阶差分，jerk) ==========
        if N >= 4:
            # d3 = P_{i+3} - 3*P_{i+2} + 3*P_{i+1} - P_i
            d3 = (
                ctrl_pts[:, :, 3:]            # P_{i+3}
                - 3.0 * ctrl_pts[:, :, 2:-1]   # -3*P_{i+2}
                + 3.0 * ctrl_pts[:, :, 1:-2]   # +3*P_{i+1}
                - ctrl_pts[:, :, :-3]          # -P_i
            )  # (B, 3, N-3)
            
            # 计算加权平方和: d3^T * R_smooth * d3
            smooth_cost = torch.einsum("bcn,cd,bdn->bn", d3, self.R_smooth, d3)  # (B, N-3)
            smoothness_cost = smooth_cost.mean(dim=1)  # (B,)
        else:
            smoothness_cost = torch.zeros(B, device=ctrl_pts.device)
        
        return smoothness_cost, acceleration_cost