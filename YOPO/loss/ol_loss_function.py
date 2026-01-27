import torch.nn as nn
from config.config import cfg
from loss.ol_safety_loss import SafetyLoss
from loss.ol_smoothness_loss import BSplineSmoothnessLoss
from loss.ol_guidance_loss import GuidanceLoss


class OLnetLoss(nn.Module):
    def __init__(self):
        super(OLnetLoss, self).__init__()

        # 权重参数
        vel_scale = cfg["vel_max_train"] / 1.0
        self.smoothness_weight = cfg["ws"] / vel_scale ** 5
        self.accele_weight = cfg["wa"] / vel_scale ** 3
        self.safety_weight = cfg["wc"]
        self.goal_weight = cfg["wg"]
        
        # 代价函数
        self.smoothness_loss = BSplineSmoothnessLoss()
        self.safety_loss = SafetyLoss()
        self.goal_loss = GuidanceLoss()

        print("------ Actual Loss ------")
        print(f"| {'smooth':<12} = {self.smoothness_weight:6.4f} |")
        print(f"| {'safety':<12} = {self.safety_weight:6.4f} |")
        print(f"| {'goal':<12} = {self.goal_weight:6.4f} |")
        print("-------------------------")

    def forward(self, w_pos, goal, map_id):
        """
        Args:
            w_pos: (batch_size, 3, N) → B-spline control points
            goal: (batch_size, 3) → goal position
            map_id: (batch_size) which ESDF map to query

        Returns:
            cost: (batch_size) → weighted cost
        """
        smoothness_cost, acceleration_cost = self.smoothness_loss(w_pos)
        safety_cost = self.safety_loss(w_pos, map_id)
        goal_cost = self.goal_loss(w_pos, goal)

        return self.smoothness_weight * smoothness_cost, self.safety_weight * safety_cost, self.goal_weight * goal_cost, self.accele_weight * acceleration_cost
        # return self.smoothness_weight * smoothness_cost
