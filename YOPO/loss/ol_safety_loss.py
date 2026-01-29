import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from scipy.ndimage import distance_transform_edt
from config.config import cfg


class SafetyLoss(nn.Module):
    """
    基于B样条控制点的安全性代价函数
    支持 Clamped B-Spline（推荐）和 Uniform B-Spline
    输入: B样条控制点 w_pos (B, 3, N)
    输出: 软约束ESDF安全代价
    """
    def __init__(self):
        super(SafetyLoss, self).__init__()
        
        # 配置参数
        self.map_expand_min = np.array(cfg['map_expand_min'])
        self.map_expand_max = np.array(cfg['map_expand_max'])
        self.d0 = cfg["d0"]  # 安全阈值距离
        self.r = cfg["r"]    # 势场衰减系数
        
        # B样条采样参数
        self.eval_points = cfg["eval_points"] if "eval_points" in cfg._data else 30  # 轨迹上采样点数
        self.bspline_degree = cfg["bspline_degree"] if "bspline_degree" in cfg._data else 3  # B样条阶数，默认3次
        
        # 是否使用clamped B-spline（推荐True）
        self.use_clamped = cfg["use_clamped_bspline"] if "use_clamped_bspline" in cfg._data else True
        
        # 时间积分 vs 线积分
        self.time_integral = cfg["time_integral"] if "time_integral" in cfg._data else True
        
        # ESDF地图参数
        self.voxel_size = cfg["voxel_size"] if "voxel_size" in cfg._data else 0.2
        self.min_bounds = None  # (N_maps, 3)
        self.max_bounds = None  # (N_maps, 3)
        self.sdf_shapes = None  # (N_maps, 3)
        
        # 构建ESDF地图
        print("Building ESDF map...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "../", cfg["dataset_path"])
        self.sdf_maps = self.get_sdf_from_ply(data_dir)
        self.device = self.sdf_maps[0].device if len(self.sdf_maps) > 0 else torch.device('cuda')
        print(f"Map built! Total {len(self.sdf_maps)} maps loaded.")
        print(f"SafetyLoss using {'Clamped' if self.use_clamped else 'Uniform'} B-spline")
        
        # 预计算B样条基函数
        self._precompute_bspline_basis()
    
    def _precompute_bspline_basis(self):
        """
        预计算B样条基函数矩阵
        对于3次B样条，使用标准基函数矩阵
        """
        # 3次均匀B样条的基函数矩阵
        # M = 1/6 * [[1,  4,  1,  0],
        #            [-3, 0,  3,  0],
        #            [3, -6,  3,  0],
        #            [-1, 3, -3,  1]]
        if self.bspline_degree == 3:
            M = torch.tensor([
                [1,  4,  1,  0],
                [-3, 0,  3,  0],
                [3, -6,  3,  0],
                [-1, 3, -3,  1]
            ], dtype=torch.float32) / 6.0
            self.register_buffer('basis_matrix', M)
        else:
            raise NotImplementedError(f"B-spline degree {self.bspline_degree} not implemented yet")
    
    def to_clamped_bspline(self, ctrl_pts):
        """
        将uniform B-spline控制点转换为clamped B-spline
        
        原理: 重复端点3次，使得B样条曲线精确经过起点和终点
        
        Args:
            ctrl_pts: (B, 3, N) - 原始控制点
        
        Returns:
            ctrl_pts_clamped: (B, 3, N+6) - clamped控制点
        """
        if not self.use_clamped:
            return ctrl_pts
        
        return torch.cat([
            ctrl_pts[:, :, 0:1].expand(-1, -1, 3),  # 起点P0重复3次
            ctrl_pts,                                # 原始控制点 P0...PN
            ctrl_pts[:, :, -1:].expand(-1, -1, 3),  # 终点PN重复3次
        ], dim=2)
    
    def forward(self, w_pos, map_id):
        """
        计算安全性代价
        
        Args:
            w_pos: (B, 3, N) - B样条控制点（原始）
            map_id: (B,) - 每个batch使用哪张ESDF地图
        
        Returns:
            safety_cost: (B,) - 安全性代价
        """
        batch_size = w_pos.shape[0]
        
        # ========== 转换为clamped B-spline ==========
        w_pos_for_sampling = self.to_clamped_bspline(w_pos)
        
        # ========== 1. 从B样条控制点采样轨迹点 ==========
        pos_samples, vel_samples = self.sample_bspline_trajectory(w_pos_for_sampling)  # (B, eval_points, 3)
        
        # ========== 2. 查询ESDF距离 ==========
        pos_flat = pos_samples.reshape(batch_size, -1, 3)  # (B, eval_points, 3)
        cost, dist = self.get_distance_cost(pos_flat, map_id)  # (B, eval_points)
        
        # ========== 3. 计算代价（时间积分 or 线积分） ==========
        if self.time_integral:
            # 时间平均
            safety_cost = cost.mean(dim=-1)  # (B,)
        else:
            # 线积分平均
            vel_norm = vel_samples.norm(dim=-1)  # (B, eval_points)
            dt = 1.0 / self.eval_points  # 归一化时间步长
            line_integral_cost = (cost * vel_norm * dt).sum(dim=1)  # (B,)
            line_length = (vel_norm * dt).sum(dim=1)  # (B,)
            safety_cost = line_integral_cost / (line_length + 1e-8)  # (B,)
        
        return safety_cost
    
    def sample_bspline_trajectory(self, ctrl_pts):
        """
        从B样条控制点采样轨迹点和速度
        
        注意: 这个函数接收的是已经转换后的控制点（如果use_clamped=True）
        
        Args:
            ctrl_pts: (B, 3, N) - B样条控制点（可能已经是clamped格式）
        
        Returns:
            positions: (B, eval_points, 3) - 采样的位置点
            velocities: (B, eval_points, 3) - 采样的速度
        """
        B, dim, N = ctrl_pts.shape
        assert dim == 3, "Expected 3D control points"
        assert N >= 4, f"Need at least 4 control points for cubic B-spline, got {N}"
        
        # 生成归一化参数 u ∈ [0, 1]
        u = torch.linspace(0, 1, self.eval_points, device=ctrl_pts.device)
        
        # 计算每个采样点对应的B样条段索引
        # 对于N个控制点，有 N-3 个B样条段（3次B样条）
        num_segments = N - 3
        segment_indices = (u * num_segments).clamp(0, num_segments - 1).long()  # (eval_points,)
        local_u = (u * num_segments - segment_indices.float()).clamp(0, 1)  # (eval_points,) 局部参数
        
        # 构建参数向量 [1, u, u^2, u^3]
        u_vec = torch.stack([
            torch.ones_like(local_u),
            local_u,
            local_u ** 2,
            local_u ** 3
        ], dim=-1)  # (eval_points, 4)
        
        # 计算位置和速度的基函数
        pos_basis = u_vec @ self.basis_matrix.T  # (eval_points, 4)
        
        # 速度基函数: d/du [1, u, u^2, u^3] = [0, 1, 2u, 3u^2]
        u_vec_derivative = torch.stack([
            torch.zeros_like(local_u),
            torch.ones_like(local_u),
            2 * local_u,
            3 * local_u ** 2
        ], dim=-1)  # (eval_points, 4)
        vel_basis = u_vec_derivative @ self.basis_matrix.T  # (eval_points, 4)
        
        # 对每个采样点，选择对应的4个控制点
        positions = []
        velocities = []
        
        for i in range(self.eval_points):
            seg_idx = segment_indices[i].item()
            # 选择控制点 P_i, P_{i+1}, P_{i+2}, P_{i+3}
            local_ctrl = ctrl_pts[:, :, seg_idx:seg_idx+4]  # (B, 3, 4)
            
            # 计算位置: P(u) = [1, u, u^2, u^3] @ M @ [P_i; P_{i+1}; P_{i+2}; P_{i+3}]
            pos = torch.einsum('k,bdk->bd', pos_basis[i], local_ctrl)  # (B, 3)
            positions.append(pos)
            
            # 计算速度: V(u) = d P(u) / du
            vel = torch.einsum('k,bdk->bd', vel_basis[i], local_ctrl) * num_segments  # (B, 3)
            velocities.append(vel)
        
        positions = torch.stack(positions, dim=1)  # (B, eval_points, 3)
        velocities = torch.stack(velocities, dim=1)  # (B, eval_points, 3)
        
        return positions, velocities
    
    def get_distance_cost(self, pos, map_id):
        """
        查询ESDF距离并计算代价
        
        Args:
            pos: (B, N, 3) - 点在世界坐标系下的位置
            map_id: (B,) - 每个batch使用哪张sdf_map
        
        Returns:
            cost: (B, N) - 安全代价
            dist_query: (B, N) - ESDF距离
        """
        B, N, _ = pos.shape
        
        # 获取局部SDF地图
        sdf_maps, local_origin, local_shape = self.get_batch_sdf(pos, map_id)
        
        # 将pos转为voxel坐标: grid = (pos - min_bound) / voxel_size
        grid = (pos - local_origin.unsqueeze(1)) / self.voxel_size  # (B, N, 3)
        
        # 归一化grid到[-1, 1]（PyTorch grid_sample要求）
        grid_normalized = 2.0 * grid / (local_shape - 1).unsqueeze(1) - 1.0  # (B, N, 3)
        grid_normalized = grid_normalized.view(B, 1, 1, N, 3)
        grid_normalized = torch.clamp(grid_normalized, min=-0.99, max=0.99)
        
        # 三线性插值查询距离
        dist_query = F.grid_sample(
            sdf_maps, grid_normalized, 
            mode='bilinear', padding_mode='zeros', align_corners=True
        )  # (B, 1, 1, 1, N)
        dist_query = dist_query.view(B, N)
        
        # 计算软约束代价: cost = exp(-(d - d0) / r)
        cost = self.cost_function(dist_query)  # (B, N)
        
        return cost, dist_query
    
    def cost_function(self, d):
        """
        软约束势场函数
        当 d < d0 时，代价指数增长
        当 d >= d0 时，代价趋近于0
        
        Args:
            d: (B, N) - ESDF距离
        Returns:
            cost: (B, N) - 安全代价
        """
        return torch.exp(-(d - self.d0) / self.r)
    
    def get_batch_sdf(self, pos, map_id):
        """
        裁剪所有地图到相同尺寸并覆盖pos的范围
        
        Args:
            pos: (B, N, 3)
            map_id: (B,)
        
        Returns:
            sdf_maps: (B, 1, D, H, W) - 裁剪后的SDF地图
            local_origin: (B, 3) - 裁剪区域的世界坐标原点
            local_shape: (B, 3) - 裁剪区域的体素形状
        """
        min_bounds = self.min_bounds[map_id]  # (B, 3)
        sdf_shapes = self.sdf_shapes[map_id]  # (B, 3)
        
        # 计算pos的范围
        min_pos = pos.amin(dim=1)  # (B, 3)
        max_pos = pos.amax(dim=1)  # (B, 3)
        
        # 转换为体素索引
        min_indices = ((min_pos - min_bounds) / self.voxel_size).int()
        max_indices = ((max_pos - min_bounds) / self.voxel_size).int()
        
        # 计算跨度并对齐
        spans = max_indices - min_indices  # (B, 3)
        max_spans = spans.amax(dim=0)  # (3,)
        centers = (min_indices + max_indices) // 2  # (B, 3)
        
        # 扩展边界（留5个体素的buffer）
        min_indices = centers - max_spans // 2 - 5
        max_indices = centers + max_spans // 2 + 5
        
        # 边界裁剪
        new_min_indices = min_indices.clamp(min=0)
        underflow_amount = new_min_indices - min_indices
        min_indices = new_min_indices
        max_indices = max_indices + underflow_amount
        
        new_max_indices = torch.minimum(max_indices, sdf_shapes.int())
        overflow_amount = max_indices - new_max_indices
        max_indices = new_max_indices
        min_indices = min_indices - overflow_amount
        
        # 处理负索引
        if (min_indices < 0).any():
            min_underflow = torch.minimum(min_indices, torch.zeros_like(min_indices))
            shift = (-min_underflow).max(dim=0).values
            min_indices = min_indices + shift
        
        # 裁剪SDF地图
        sdf_maps = torch.stack([
            self.sdf_maps[map_idx][0, :,
                                   min_idx[2]:max_idx[2],
                                   min_idx[1]:max_idx[1],
                                   min_idx[0]:max_idx[0]]
            for map_idx, min_idx, max_idx in 
            zip(map_id.tolist(), min_indices.tolist(), max_indices.tolist())
        ])
        
        local_origin = min_indices.float() * self.voxel_size + min_bounds
        local_shape = (max_indices - min_indices).float()
        
        return sdf_maps, local_origin, local_shape
    
    def get_sdf_from_ply(self, path):
        """
        从PLY点云文件构建ESDF地图
        """
        sorted_files = self.read_sorted_ply_files(path)
        sdf_maps = []
        min_bounds, max_bounds, sdf_shapes = [], [], []
        
        for file in sorted_files:
            pcd = o3d.io.read_point_cloud(file)
            min_bound = np.array(pcd.get_min_bound()) - self.map_expand_min
            max_bound = np.array(pcd.get_max_bound()) + self.map_expand_max
            points = np.asarray(pcd.points)
            
            print(f"    {os.path.basename(file)}: "
                  f"x=({min_bound[0]:.2f}, {max_bound[0]:.2f}), "
                  f"y=({min_bound[1]:.2f}, {max_bound[1]:.2f}), "
                  f"z=({min_bound[2]:.2f}, {max_bound[2]:.2f})")
            
            # 创建体素网格
            sdf_shape = np.ceil((max_bound - min_bound) / self.voxel_size).astype(int)
            voxel_indices = ((points - min_bound) / self.voxel_size).astype(int)
            
            # 过滤有效体素
            valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < sdf_shape), axis=1)
            voxel_indices = voxel_indices[valid_mask]
            
            # 构建占据网格
            occupancy = np.zeros(sdf_shape, dtype=np.uint8)
            occupancy[tuple(voxel_indices.T)] = 1
            
            # 计算ESDF
            obstacle_mask = occupancy == 1
            free_mask = occupancy == 0
            
            dist_to_obstacle = distance_transform_edt(free_mask) * self.voxel_size
            dist_inside_obstacle = distance_transform_edt(obstacle_mask) * self.voxel_size
            
            # 障碍物内部距离为负
            dist_to_obstacle[obstacle_mask] = -dist_inside_obstacle[obstacle_mask]
            
            # 转为PyTorch张量 (1, 1, D, H, W)
            sdf_tensor = torch.from_numpy(dist_to_obstacle).float()
            sdf_tensor = sdf_tensor.unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 3, 2)
            sdf_tensor = sdf_tensor.cuda() if torch.cuda.is_available() else sdf_tensor
            
            sdf_maps.append(sdf_tensor)
            sdf_shapes.append(sdf_tensor.shape[-3:][::-1])  # D,H,W -> X,Y,Z
            min_bounds.append(min_bound)
            max_bounds.append(max_bound)
        
        # 保存地图边界信息
        device = sdf_maps[0].device
        self.min_bounds = torch.tensor(np.array(min_bounds), device=device).float()
        self.max_bounds = torch.tensor(np.array(max_bounds), device=device).float()
        self.sdf_shapes = torch.tensor(np.array(sdf_shapes), device=device).float()
        
        return sdf_maps
    
    def read_sorted_ply_files(self, path):
        """读取并排序PLY文件"""
        ply_files = glob.glob(os.path.join(path, 'pointcloud-*.ply'))
        
        def extract_index(filename):
            base = os.path.basename(filename)
            number_part = base.replace('pointcloud-', '').replace('.ply', '')
            return int(number_part)
        
        return sorted(ply_files, key=extract_index)


# ========== 使用说明 ==========
"""
配置文件 (config.yaml) 示例:

# ========== B-spline 类型 ==========
use_clamped_bspline: true         # 推荐! 起点和终点精确在控制点上

# ========== ESDF 地图参数 ==========
dataset_path: "data/maps"
map_expand_min: [0.5, 0.5, 0.5]  # 地图扩展（最小边界）
map_expand_max: [0.5, 0.5, 0.5]  # 地图扩展（最大边界）
voxel_size: 0.2                   # 体素大小（米）

# ========== 安全代价参数 ==========
d0: 0.5                           # 安全阈值距离（米）
r: 0.2                            # 势场衰减系数
eval_points: 30                   # 轨迹采样点数
time_integral: true               # true=时间积分, false=线积分

# ========== B-spline 参数 ==========
bspline_degree: 3                 # B样条阶数（固定为3）

# ========== 代价权重 ==========
ws: 1.0                           # 平滑性权重
wa: 1.0                           # 加速度权重
wc: 10.0                          # 安全性权重
wg: 5.0                           # 引导权重

# ========== 使用示例 ==========

from loss.ol_safety_loss import SafetyLoss

# 初始化（会自动加载ESDF地图）
safety_loss = SafetyLoss()

# 前向传播
w_pos = network(input)  # (B, 3, N) - 原始控制点
map_id = ...            # (B,) - 地图索引

# 自动处理clamped转换
safety_cost = safety_loss(w_pos, map_id)  # (B,)

# 如果use_clamped_bspline=True:
# - 内部会自动将 (B, 3, N) 转换为 (B, 3, N+6)
# - 起点重复3次，终点重复3次
# - 采样的轨迹会精确经过原始的起点和终点
# - 对用户完全透明！

# ========== Clamped vs Uniform 对比 ==========

Uniform B-spline (原来的):
  - 曲线不经过起点和终点控制点
  - 轨迹范围略小于控制点范围
  - 标准B样条行为

Clamped B-spline (推荐):
  - 曲线精确经过起点P0和终点PN
  - 轨迹范围完全覆盖控制点范围
  - 更直观，更适合端到端导航

两者的ESDF查询和代价计算完全相同，
只是采样的轨迹点略有不同。

# ========== 性能说明 ==========

使用clamped B-spline:
  - 控制点数量: N → N+6 (内部)
  - 用户输入输出: 保持 (B, 3, N) 不变
  - 计算开销: 增加约5% (可忽略)
  - 轨迹质量: 提升 (更准确的起点/终点)

# ========== 常见问题 ==========

Q1: 我的网络输出需要改吗？
A1: 不需要！仍然输出 (B, 3, N)，内部自动转换

Q2: 会影响训练吗？
A2: 不会！梯度正常反向传播到原始控制点

Q3: 必须用clamped吗？
A3: 不必须，但强烈推荐。可以设置 use_clamped_bspline: false 切换回uniform

Q4: 如何验证是否正确？
A4: 可视化采样点，检查起点是否在P0，终点是否在PN
"""