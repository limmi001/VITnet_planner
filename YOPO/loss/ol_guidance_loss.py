import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import cfg


class GuidanceLoss(nn.Module):
    """
    Clamped B-Spline 引导代价函数
    目标: 让轨迹终点方向尽量沿着起点→目标点的方向
    
    优势: 起点=P0, 终点=PN (精确!)
    """
    def __init__(self):
        super(GuidanceLoss, self).__init__()
        self.goal_length = cfg.get('goal_length', 1.0)
        self.vel_dir_weight = cfg.get('vel_dir_weight', 0)  # 终点速度方向约束权重
        
        # 损失类型选择: 'distance' 或 'similarity'
        self.loss_type = cfg.get('guidance_loss_type', 'similarity')
        
        # 垂直方向惩罚权重（只在similarity模式下使用）
        self.perp_weight = cfg.get('perp_weight', 0.5)
        
        # 是否使用clamped B-spline（推荐True）
        # self.use_clamped = cfg.get('use_clamped_bspline', True)
        self.use_clamped = True
        
        print(f"GuidanceLoss initialized:")
        print(f"  - Type: {self.loss_type}")
        print(f"  - Clamped B-spline: {self.use_clamped}")
        print(f"  - Perp weight: {self.perp_weight}")
        print(f"  - Vel direction weight: {self.vel_dir_weight}")
    
    def forward(self, w_pos, goal, start_pos=None):
        """
        Args:
            w_pos: (B, 3, N) - B样条控制点
            goal: (B, 3) - 目标位置
            start_pos: (B, 3) - 起始位置（可选，如果不提供则使用第一个控制点）
        
        Returns:
            guidance_loss: (B,) - 引导代价
        """
        B, dim, N = w_pos.shape
        assert dim == 3, "Expected 3D control points"
        
        # ========== 1. 计算起点和终点位置 ==========
        if self.use_clamped:
            # Clamped B-spline
            cur_pos = w_pos[:, :, 0] if start_pos is None else start_pos
            end_pos = w_pos[:, :, -1]
        else:
            # Uniform B-spline: 需要使用加权平均公式
            if start_pos is None:
                if N >= 3:
                    cur_pos = (w_pos[:, :, 0] + 4 * w_pos[:, :, 1] + w_pos[:, :, 2]) / 6
                else:
                    cur_pos = w_pos[:, :, 0]
            else:
                cur_pos = start_pos
            
            if N >= 4:
                end_pos = (w_pos[:, :, -3] + 4 * w_pos[:, :, -2] + w_pos[:, :, -1]) / 6
            else:
                end_pos = w_pos[:, :, -1]
        
        # ========== 2. 计算方向向量 ==========
        traj_dir = end_pos - cur_pos  # 实际轨迹方向: 起点 -> 终点
        goal_dir = goal - cur_pos     # 期望方向: 起点 -> 目标
        
        # ========== 3. 选择损失类型 ==========
        if self.loss_type == 'distance':
            guidance_loss = self.distance_loss(traj_dir, goal_dir)
        else:  # 'similarity'
            guidance_loss = self.similarity_loss(traj_dir, goal_dir)
        
        # ========== 4. 终点速度方向约束（可选） ==========
        if self.vel_dir_weight > 0 and N >= 2:
            # 计算终点速度: V(1) = 3 * (P_N - P_{N-1})
            # 注意: 不管是否clamped，速度公式都一样
            end_vel = 3.0 * (w_pos[:, :, -1] - w_pos[:, :, -2])
            vel_dir_loss = self.derivative_similarity_loss(end_vel, goal_dir)
            guidance_loss = guidance_loss + self.vel_dir_weight * vel_dir_loss
        
        return guidance_loss
    
    def distance_loss(self, traj_dir, goal_dir):
        """
        距离损失: L1距离，更直接到达目标
        
        优点: 轨迹更直，精度更高
        缺点: 在大场景下速度略慢
        
        Returns:
            l1_distance: (B,) - L1距离
        """
        l1_distance = F.smooth_l1_loss(traj_dir, goal_dir, reduction='none')  # (B, 3)
        l1_distance = l1_distance.sum(dim=1)  # (B,)
        return l1_distance
    
    def similarity_loss(self, traj_dir, goal_dir):
        """
        相似度损失: 投影长度 + 垂直偏差
        
        优点: 在大场景下飞行速度更快，允许适度的侧向探索
        缺点: 轨迹可能不够直
        
        核心思想:
        - 鼓励轨迹沿着目标方向的投影长度
        - 惩罚垂直于目标方向的偏差（权重可调）
        
        Returns:
            similarity_loss: (B,)
        """
        # 归一化目标方向
        goal_dir_norm = goal_dir / (goal_dir.norm(dim=1, keepdim=True) + 1e-8)  # (B, 3)
        
        # 轨迹在目标方向上的投影长度
        traj_along = (traj_dir * goal_dir_norm).sum(dim=1)  # (B,)
        
        # 目标距离
        goal_length = goal_dir.norm(dim=1)  # (B,)
        
        # 沿目标方向的长度差异（余弦相似度）
        parallel_diff = F.smooth_l1_loss(goal_length, traj_along, reduction='none')  # (B,)
        
        # 垂直于目标方向的长度
        traj_perp = traj_dir - traj_along.unsqueeze(1) * goal_dir_norm  # (B, 3)
        perp_diff = traj_perp.norm(dim=1)  # (B,)
        
        # 组合损失（perp_weight控制垂直约束强度）
        # perp_weight = 0: 只关心沿目标方向的长度，允许自由横向移动（速度快）
        # perp_weight = 1: 等价于distance_loss，要求精确到达（更直）
        similarity_loss = parallel_diff + self.perp_weight * perp_diff
        
        return similarity_loss
    
    def derivative_similarity_loss(self, derivative, goal_dir):
        """
        速度方向约束: 让终点速度方向朝向目标
        
        Args:
            derivative: (B, 3) - 终点速度
            goal_dir: (B, 3) - 目标方向
        
        Returns:
            vel_similarity_loss: (B,) - 速度方向损失 (0表示完全对齐，2表示反向)
        """
        # 归一化
        goal_dir_norm = goal_dir / (goal_dir.norm(dim=1, keepdim=True) + 1e-8)  # (B, 3)
        derivative_norm = derivative / (derivative.norm(dim=1, keepdim=True) + 1e-8)  # (B, 3)
        
        # 余弦相似度: cos(θ) ∈ [-1, 1]
        similarity = (derivative_norm * goal_dir_norm).sum(dim=1)  # (B,)
        
        # 转为损失: 1 - cos(θ)
        # cos(θ)=1 (对齐) → loss=0
        # cos(θ)=0 (垂直) → loss=1
        # cos(θ)=-1 (反向) → loss=2
        return 1 - similarity


'''
包含速度方向约束和可调垂直权重的高级版本（下次实验试一试）
'''
# class GuidanceLossAdvanced(nn.Module):
#     """
#     高级版本: 支持更多控制选项
#     - 自适应垂直权重（根据距离目标远近调整）
#     - 中间点方向约束（让整条轨迹都朝向目标）
#     """
#     def __init__(self):
#         super(GuidanceLossAdvanced, self).__init__()
#         self.goal_length = cfg.get('goal_length', 1.0)
#         self.vel_dir_weight = cfg.get('vel_dir_weight', 0)
#         self.loss_type = cfg.get('guidance_loss_type', 'similarity')
#         self.perp_weight = cfg.get('perp_weight', 0.5)
        
#         # 是否使用clamped B-spline
#         self.use_clamped = cfg.get('use_clamped_bspline', True)
        
#         # 额外选项
#         self.use_adaptive_perp_weight = cfg.get('use_adaptive_perp_weight', False)
#         self.use_intermediate_points = cfg.get('use_intermediate_points', False)
#         self.n_intermediate = cfg.get('n_intermediate_points', 3)
        
#         print(f"GuidanceLossAdvanced initialized:")
#         print(f"  - Type: {self.loss_type}")
#         print(f"  - Clamped B-spline: {self.use_clamped}")
#         print(f"  - Adaptive perp weight: {self.use_adaptive_perp_weight}")
#         print(f"  - Intermediate points: {self.use_intermediate_points}")
    
#     def forward(self, w_pos, goal, start_pos=None):
#         """
#         Args:
#             w_pos: (B, 3, N) - B样条控制点
#             goal: (B, 3) - 目标位置
#             start_pos: (B, 3) - 起始位置
        
#         Returns:
#             guidance_loss: (B,)
#         """
#         B, dim, N = w_pos.shape
        
#         # ========== 计算起点和终点 ==========
#         if self.use_clamped:
#             # Clamped B-spline: 简单！
#             cur_pos = w_pos[:, :, 0] if start_pos is None else start_pos
#             end_pos = w_pos[:, :, -1]
#         else:
#             # Uniform B-spline: 需要加权平均
#             if start_pos is None:
#                 if N >= 3:
#                     cur_pos = (w_pos[:, :, 0] + 4 * w_pos[:, :, 1] + w_pos[:, :, 2]) / 6
#                 else:
#                     cur_pos = w_pos[:, :, 0]
#             else:
#                 cur_pos = start_pos
            
#             if N >= 4:
#                 end_pos = (w_pos[:, :, -3] + 4 * w_pos[:, :, -2] + w_pos[:, :, -1]) / 6
#             else:
#                 end_pos = w_pos[:, :, -1]
        
#         # 方向向量
#         traj_dir = end_pos - cur_pos
#         goal_dir = goal - cur_pos
        
#         # ========== 自适应垂直权重（可选） ==========
#         if self.use_adaptive_perp_weight:
#             # 距离远时，允许更多侧向探索（低perp_weight）
#             # 距离近时，要求更精确（高perp_weight）
#             distance_to_goal = goal_dir.norm(dim=1, keepdim=True)  # (B, 1)
#             adaptive_perp_weight = torch.clamp(
#                 1.0 - distance_to_goal / (self.goal_length + 1e-8),
#                 min=0.1,
#                 max=1.0
#             )
#             original_perp_weight = self.perp_weight
#             self.perp_weight = adaptive_perp_weight.mean().item()
        
#         # ========== 基础损失 ==========
#         if self.loss_type == 'distance':
#             guidance_loss = self.distance_loss(traj_dir, goal_dir)
#         else:
#             guidance_loss = self.similarity_loss(traj_dir, goal_dir)
        
#         # 恢复原始权重
#         if self.use_adaptive_perp_weight:
#             self.perp_weight = original_perp_weight
        
#         # ========== 速度方向约束 ==========
#         if self.vel_dir_weight > 0 and N >= 2:
#             end_vel = 3.0 * (w_pos[:, :, -1] - w_pos[:, :, -2])
#             vel_dir_loss = self.derivative_similarity_loss(end_vel, goal_dir)
#             guidance_loss = guidance_loss + self.vel_dir_weight * vel_dir_loss
        
#         # ========== 中间点方向约束（可选） ==========
#         if self.use_intermediate_points and N >= self.n_intermediate + 2:
#             intermediate_loss = self.intermediate_direction_loss(w_pos, cur_pos, goal_dir)
#             guidance_loss = guidance_loss + 0.1 * intermediate_loss  # 较小权重
        
#         return guidance_loss
    
#     def distance_loss(self, traj_dir, goal_dir):
#         """距离损失"""
#         l1_distance = F.smooth_l1_loss(traj_dir, goal_dir, reduction='none')
#         return l1_distance.sum(dim=1)
    
#     def similarity_loss(self, traj_dir, goal_dir):
#         """相似度损失"""
#         goal_dir_norm = goal_dir / (goal_dir.norm(dim=1, keepdim=True) + 1e-8)
#         traj_along = (traj_dir * goal_dir_norm).sum(dim=1)
#         goal_length = goal_dir.norm(dim=1)
#         parallel_diff = F.smooth_l1_loss(goal_length, traj_along, reduction='none')
#         traj_perp = traj_dir - traj_along.unsqueeze(1) * goal_dir_norm
#         perp_diff = traj_perp.norm(dim=1)
#         return parallel_diff + self.perp_weight * perp_diff
    
#     def derivative_similarity_loss(self, derivative, goal_dir):
#         """速度方向损失"""
#         goal_dir_norm = goal_dir / (goal_dir.norm(dim=1, keepdim=True) + 1e-8)
#         derivative_norm = derivative / (derivative.norm(dim=1, keepdim=True) + 1e-8)
#         similarity = (derivative_norm * goal_dir_norm).sum(dim=1)
#         return 1 - similarity
    
#     def intermediate_direction_loss(self, w_pos, start_pos, goal_dir):
#         """
#         中间点方向约束: 让轨迹的多个中间控制点也朝向目标
        
#         Args:
#             w_pos: (B, 3, N)
#             start_pos: (B, 3)
#             goal_dir: (B, 3)
#         """
#         B, dim, N = w_pos.shape
#         goal_dir_norm = goal_dir / (goal_dir.norm(dim=1, keepdim=True) + 1e-8)
        
#         # 选择N个均匀分布的中间控制点
#         indices = torch.linspace(1, N-2, self.n_intermediate, device=w_pos.device).long()
        
#         total_loss = 0
#         for idx in indices:
#             # 中间点到起点的方向
#             intermediate_pos = w_pos[:, :, idx]
#             intermediate_dir = intermediate_pos - start_pos
#             intermediate_dir_norm = intermediate_dir / (intermediate_dir.norm(dim=1, keepdim=True) + 1e-8)
            
#             # 与目标方向的相似度
#             similarity = (intermediate_dir_norm * goal_dir_norm).sum(dim=1)
#             total_loss = total_loss + (1 - similarity)
        
#         return total_loss / self.n_intermediate


# ========== 使用示例和配置说明 ==========
"""
配置文件 (config.yaml) 示例:

# ========== 基础配置 ==========
use_clamped_bspline: true         # 强烈推荐! 起点和终点精确在控制点上

# ========== 引导损失配置 ==========
guidance_loss_type: 'similarity'  # 'distance' or 'similarity'
perp_weight: 0.5                  # 垂直方向惩罚权重 [0.0-1.0]
vel_dir_weight: 0                 # 终点速度方向约束权重 (0表示不使用)
goal_length: 10.0                 # 典型目标距离（用于归一化）

# ========== 高级选项（GuidanceLossAdvanced） ==========
use_adaptive_perp_weight: false   # 自适应调整垂直权重
use_intermediate_points: false    # 使用中间点方向约束
n_intermediate_points: 3          # 中间点数量

# ========== 使用建议 ==========

1. 推荐配置（大多数场景）:
   use_clamped_bspline: true
   guidance_loss_type: 'similarity'
   perp_weight: 0.5
   vel_dir_weight: 0
   
2. 大场景 + 追求速度:
   guidance_loss_type: 'similarity'
   perp_weight: 0.0~0.3
   
3. 小场景 + 追求精度:
   guidance_loss_type: 'distance'
   或 guidance_loss_type: 'similarity', perp_weight: 0.8~1.0
   
4. 需要平滑到达目标:
   vel_dir_weight: 5~10
   
5. 复杂障碍场景（使用高级版本）:
   use_intermediate_points: true
   n_intermediate_points: 3

# ========== Clamped vs Uniform 对比 ==========

Uniform B-spline (原来的):
  - 起点计算: (P0 + 4*P1 + P2) / 6  ← 复杂
  - 终点计算: (P_{N-3} + 4*P_{N-2} + P_{N-1}) / 6  ← 复杂
  - 优点: 标准B样条
  - 缺点: 不直观，端点不在控制点上

Clamped B-spline (推荐):
  - 起点计算: P0  ← 简单！
  - 终点计算: PN  ← 简单！
  - 优点: 直观，端点精确，代码简洁
  - 缺点: 无（推荐使用）

# ========== 代码使用示例 ==========

from loss.ol_guidance_loss import GuidanceLoss

# 初始化
guidance_loss = GuidanceLoss()

# 前向传播
w_pos = network(input)  # (B, 3, N)
goal = ...              # (B, 3)
start_pos = ...         # (B, 3) 可选

goal_cost = guidance_loss(w_pos, goal, start_pos)  # (B,)

# 如果使用clamped (推荐):
# - 起点就是w_pos[:, :, 0]
# - 终点就是w_pos[:, :, -1]
# - 不需要复杂的加权平均公式！
"""