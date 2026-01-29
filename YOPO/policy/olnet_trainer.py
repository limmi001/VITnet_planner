"""
OLNet Training Strategy
Supervised learning based on B-spline control points
Adapted from YOPO trainer but uses OLNetwork and OLnetLoss
"""
import os
import time
import atexit
import torch
import numpy as np
from torch.nn import functional as F
from rich.progress import Progress
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from config.config import cfg
from loss.ol_loss_function import OLnetLoss
from policy.ol_network import OLNetwork
from policy.yopo_dataset import YOPODataset  # 复用原数据集
from policy.state_transform import *


class OLNetTrainer:
    """
    Trainer for OLNetwork
    Uses B-spline control point prediction instead of endpoint states
    """
    def __init__(
            self,
            learning_rate=0.001,
            batch_size=32,
            loss_weight=None,  # 不需要了，权重在OLnetLoss里
            tensorboard_path=None,
            checkpoint_path=None,
            save_on_exit=False,
    ):
        self.batch_size = batch_size
        self.max_grad_norm = 1.0  # 梯度裁剪阈值
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if save_on_exit: 
            self._exit_func = atexit.register(self.save_model)
        
        # logger
        self.progress_log = Progress()
        self.tensorboard_path = self.get_next_log_path(tensorboard_path)
        self.tensorboard_log = SummaryWriter(log_dir=self.tensorboard_path)
        
        # params
        self.num_ctrl_pts = cfg.get('num_ctrl_pts', 20)  # B样条控制点数量
        
        # network
        print("Loading OLNetwork...")
        self.policy = OLNetwork()
        self.policy = self.policy.to(self.device)
        
        # 加载checkpoint
        if checkpoint_path is not None:
            try:
                state_dict = torch.load(checkpoint_path, weights_only=True, map_location=self.device)
                self.policy.load_state_dict(state_dict)
                print(f"Checkpoint {checkpoint_path} loaded successfully")
            except FileNotFoundError:
                print("Checkpoint not found, training from scratch")
        else:
            print("Training from scratch")
        
        # loss function
        print("Initializing OLnetLoss...")
        self.olnet_loss = OLnetLoss()
        
        # optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-4,
            fused=True if torch.cuda.is_available() else False
        )
        
        # learning rate scheduler (可选)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=100,
            eta_min=learning_rate * 0.01
        )
        
        print("Network Loaded! Loading Dataset...")
        
        # dataset - 复用YOPODataset
        self.train_dataloader = DataLoader(
            YOPODataset(mode='train'), 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4, 
            pin_memory=True,
            drop_last=True  # 确保batch size一致
        )
        self.val_dataloader = DataLoader(
            YOPODataset(mode='valid'), 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=4, 
            pin_memory=True,
            drop_last=True
        )
        print("Dataset Loaded!")
        print(f"Training samples: {len(self.train_dataloader.dataset)}")
        print(f"Validation samples: {len(self.val_dataloader.dataset)}")
    
    def train(self, epoch, save_interval=None):
        """
        Main training loop
        """
        with self.progress_log:
            total_progress = self.progress_log.add_task("Training", total=epoch)
            
            for self.epoch_i in range(epoch):
                # Training phase
                self.policy.train()
                self.train_one_epoch(self.epoch_i, total_progress)
                
                # Validation phase
                self.policy.eval()
                self.eval_one_epoch(self.epoch_i)
                
                # Update learning rate
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.tensorboard_log.add_scalar("Train/LearningRate", current_lr, self.epoch_i)
                
                # Save checkpoint
                if save_interval is not None and (self.epoch_i + 1) % save_interval == 0:
                    self.progress_log.console.log(f"Saving model at epoch {self.epoch_i + 1}...")
                    policy_path = os.path.join(self.tensorboard_path, f"epoch_{self.epoch_i + 1}.pth")
                    torch.save(self.policy.state_dict(), policy_path)
            
            self.progress_log.console.log("Training OLNet Finished!")
            self.progress_log.remove_task(total_progress)
    
    def train_one_epoch(self, epoch: int, total_progress):
        """
        Train for one epoch
        核心训练逻辑 - 保持与YOPO相似的结构
        """
        one_epoch_progress = self.progress_log.add_task(
            f"Epoch: {epoch}", 
            total=len(self.train_dataloader)
        )
        
        inspect_interval = max(1, len(self.train_dataloader) // 16)
        
        # 损失统计
        total_losses = []
        smooth_losses = []
        safety_losses = []
        goal_losses = []
        acc_losses = []
        start_time = time.time()
        
        for step, (depth, pos, rot, obs_b, map_id) in enumerate(self.train_dataloader):
            # 数据移到GPU
            depth = depth.to(self.device)
            pos = pos.to(self.device)
            rot = rot.to(self.device)
            obs_b = obs_b.to(self.device)
            map_id = map_id.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播和损失计算
            smooth_cost, safety_cost, goal_cost, acc_cost = self.forward_and_compute_loss(
                depth, pos, rot, obs_b, map_id
            )
            
            # 总损失
            total_loss = smooth_cost + safety_cost + goal_cost + acc_cost
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            
            # 优化器步进
            self.optimizer.step()
            
            # 记录损失
            total_losses.append(total_loss.item())
            smooth_losses.append(smooth_cost.item())
            safety_losses.append(safety_cost.item())
            goal_losses.append(goal_cost.item())
            acc_losses.append(acc_cost.item())
            
            # 定期打印和记录
            if step % inspect_interval == inspect_interval - 1:
                batch_fps = inspect_interval / (time.time() - start_time)
                
                self.progress_log.console.log(
                    f"Epoch: {epoch}, "
                    f"Total Loss: {np.mean(total_losses):.4f}, "
                    f"Smooth: {np.mean(smooth_losses):.4f}, "
                    f"Safety: {np.mean(safety_losses):.4f}, "
                    f"Goal: {np.mean(goal_losses):.4f}, "
                    f"Batch FPS: {batch_fps:.2f}"
                )
                
                # TensorBoard记录
                global_step = epoch * len(self.train_dataloader) + step
                self.tensorboard_log.add_scalar("Train/TotalLoss", np.mean(total_losses), global_step)
                self.tensorboard_log.add_scalar("Train/SmoothLoss", np.mean(smooth_losses), global_step)
                self.tensorboard_log.add_scalar("Train/SafetyLoss", np.mean(safety_losses), global_step)
                self.tensorboard_log.add_scalar("Train/GoalLoss", np.mean(goal_losses), global_step)
                self.tensorboard_log.add_scalar("Train/AccelLoss", np.mean(acc_losses), global_step)
                
                # 重置统计
                total_losses, smooth_losses, safety_losses, goal_losses, acc_losses = [], [], [], [], []
                start_time = time.time()
            
            # 更新进度条
            self.progress_log.update(one_epoch_progress, advance=1)
            self.progress_log.update(total_progress, advance=1 / len(self.train_dataloader))
        
        self.progress_log.remove_task(one_epoch_progress)
    
    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int):
        """
        Evaluate for one epoch
        """
        one_epoch_progress = self.progress_log.add_task(
            f"Eval: {epoch}", 
            total=len(self.val_dataloader)
        )
        
        total_losses = []
        smooth_losses = []
        safety_losses = []
        goal_losses = []
        acc_losses = []
        
        for step, (depth, pos, rot, obs_b, map_id) in enumerate(self.val_dataloader):
            # 数据移到GPU
            depth = depth.to(self.device)
            pos = pos.to(self.device)
            rot = rot.to(self.device)
            obs_b = obs_b.to(self.device)
            map_id = map_id.to(self.device)
            
            # 前向传播和损失计算
            smooth_cost, safety_cost, goal_cost, acc_cost = self.forward_and_compute_loss(
                depth, pos, rot, obs_b, map_id
            )
            
            total_loss = smooth_cost + safety_cost + goal_cost + acc_cost
            
            # 记录损失
            total_losses.append(total_loss.item())
            smooth_losses.append(smooth_cost.item())
            safety_losses.append(safety_cost.item())
            goal_losses.append(goal_cost.item())
            acc_losses.append(acc_cost.item())
            
            self.progress_log.update(one_epoch_progress, advance=1)
        
        # 打印验证结果
        self.progress_log.console.log(
            f"Eval: {epoch}, "
            f"Total Loss: {np.mean(total_losses):.4f}, "
            f"Smooth: {np.mean(smooth_losses):.4f}, "
            f"Safety: {np.mean(safety_losses):.4f}, "
            f"Goal: {np.mean(goal_losses):.4f}"
        )
        
        # TensorBoard记录
        self.tensorboard_log.add_scalar("Eval/TotalLoss", np.mean(total_losses), epoch)
        self.tensorboard_log.add_scalar("Eval/SmoothLoss", np.mean(smooth_losses), epoch)
        self.tensorboard_log.add_scalar("Eval/SafetyLoss", np.mean(safety_losses), epoch)
        self.tensorboard_log.add_scalar("Eval/GoalLoss", np.mean(goal_losses), epoch)
        self.tensorboard_log.add_scalar("Eval/AccelLoss", np.mean(acc_losses), epoch)
        
        self.progress_log.remove_task(one_epoch_progress)
    
    def forward_and_compute_loss(self, depth, pos, rot, obs_b, map_id):
        """
        Forward pass and loss computation
        
        关键改变：
        1. 网络输出B样条控制点 w_pos (batch, 3, N)
        2. 从obs_b中提取goal
        3. 直接计算B样条轨迹损失
        
        Args:
            depth: (batch, 1, H, W) 深度图像
            pos: (batch, 3) 当前位置（世界坐标系）
            rot: (batch, 3, 3) 旋转矩阵 R_WB
            obs_b: (batch, 9) 观测 [vel_b(3), acc_b(3), goal_b(3)]
            map_id: (batch,) 地图索引
        
        Returns:
            smooth_cost: 平滑性代价
            safety_cost: 安全性代价
            goal_cost: 目标到达代价
            acc_cost: 加速度代价
        """
        batch_size = depth.shape[0]
        
        # 1. 提取目标点（机体坐标系 → 世界坐标系）
        goal_b = obs_b[:, 6:9]  # (batch, 3)
        goal_w = state_body2world_position(pos, rot, goal_b)  # (batch, 3)
        
        # 2. 网络前向传播
        # 输出：w_pos (batch, 3, N) B样条控制点（世界坐标系）
        w_pos = self.policy(depth, obs_b)  # (batch, 3, N)
        
        # 3. 将控制点转换到世界坐标系
        # 注意：OLNetwork输出的是相对于当前位置的控制点（机体坐标系）
        # 需要转换到世界坐标系
        w_pos_world = self.transform_control_points_to_world(w_pos, pos, rot)
        
        # 4. 计算损失
        smooth_cost, safety_cost, goal_cost, acc_cost = self.olnet_loss(
            w_pos_world,  # (batch, 3, N)
            goal_w,       # (batch, 3)
            map_id        # (batch,)
        )
        
        # 5. 返回平均损失
        return smooth_cost.mean(), safety_cost.mean(), goal_cost.mean(), acc_cost.mean()
    
    def transform_control_points_to_world(self, w_pos_body, pos, rot):
        """
        将B样条控制点从机体坐标系转换到世界坐标系
        
        Args:
            w_pos_body: (batch, 3, N) 机体坐标系下的控制点
            pos: (batch, 3) 当前位置
            rot: (batch, 3, 3) 旋转矩阵 R_WB
        
        Returns:
            w_pos_world: (batch, 3, N) 世界坐标系下的控制点
        """
        batch_size, _, N = w_pos_body.shape
        
        # 方法1：逐点转换
        w_pos_world = []
        for i in range(N):
            # 取出第i个控制点 (batch, 3)
            ctrl_pt_body = w_pos_body[:, :, i]
            
            # 转换到世界坐标系
            # pos_world = R_WB^T @ pos_body + current_pos
            ctrl_pt_world = state_body2world_position(pos, rot, ctrl_pt_body)
            
            w_pos_world.append(ctrl_pt_world.unsqueeze(-1))  # (batch, 3, 1)
        
        w_pos_world = torch.cat(w_pos_world, dim=-1)  # (batch, 3, N)
        
        return w_pos_world
    
    def save_model(self):
        """
        Save model checkpoint
        """
        if hasattr(self, "epoch_i"):
            self.progress_log.console.log("Saving model...")
            policy_path = os.path.join(self.tensorboard_path, f"epoch_{self.epoch_i + 1}.pth")
            torch.save(self.policy.state_dict(), policy_path)
            if hasattr(self, "_exit_func"):
                atexit.unregister(self._exit_func)
    
    def get_next_log_path(self, base_path):
        """
        Get next available log directory path
        """
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        
        nums = [int(name.split("_")[1])
                for name in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, name)) 
                and name.startswith("OLNET_") 
                and name.split("_")[1].isdigit()]
        
        next_n = max(nums, default=-1) + 1
        next_path = os.path.join(base_path, f"OLNET_{next_n}")
        os.makedirs(next_path, exist_ok=False)
        print(f"Recording tensorboard log to {next_path}")
        return next_path


# Helper function for coordinate transformation
def state_body2world_position(pos, rot, pos_body):
    """
    Transform position from body frame to world frame
    
    Args:
        pos: (batch, 3) current position in world frame
        rot: (batch, 3, 3) rotation matrix R_WB (world to body)
        pos_body: (batch, 3) position in body frame
    
    Returns:
        pos_world: (batch, 3) position in world frame
    """
    # R_BW = R_WB^T
    rot_BW = rot.transpose(-2, -1)
    
    # pos_world = R_BW @ pos_body + current_pos
    pos_world = torch.bmm(rot_BW, pos_body.unsqueeze(-1)).squeeze(-1) + pos
    
    return pos_world


if __name__ == '__main__':
    """
    测试训练器
    """
    # 创建训练器
    trainer = OLNetTrainer(
        learning_rate=0.001,
        batch_size=16,
        tensorboard_path="./logs",
        checkpoint_path=None,
        save_on_exit=True
    )
    
    # 开始训练
    trainer.train(
        epoch=100,
        save_interval=10
    )