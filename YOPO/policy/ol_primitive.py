import torch
import math


class AnnulusYawPrimitive:
    """
    Primitive definition:
    - Body frame
    - Horizontal yaw ray (XY plane only)
    - Concentric annulus (ring) primitives
    """

    _instance = None

    def __init__(
        self,
        ring_num=20,
        ring_step=0.2,
        device=None
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.ring_num = ring_num
        self.ring_step = ring_step

        # ring radius boundaries: [r0, r1, ..., rN]
        self.ring_radius = torch.arange(
            0,
            ring_num + 1,
            device=device,
            dtype=torch.float32
        ) * ring_step

        # ring center radius (anchor)
        self.ring_center = 0.5 * (self.ring_radius[:-1] + self.ring_radius[1:])

    # ------------------------------------------------------------------
    # Core geometry
    # ------------------------------------------------------------------

    def yaw_ray_direction(self, yaw):
        """
        yaw: scalar (rad)
        return: direction vector [3]
        """
        return torch.tensor(
            [math.cos(yaw), math.sin(yaw), 0.0],
            dtype=torch.float32,
            device=self.device
        )

    def get_anchor_point(self, ring_id, yaw):
        """
        Anchor point of a ring primitive:
        intersection of yaw ray and ring center radius
        """
        assert 0 <= ring_id < self.ring_num

        d = self.yaw_ray_direction(yaw)
        r = self.ring_center[ring_id]

        return r * d  # [3]

    def get_annulus_region(self, ring_id):
        """
        Return the radial interval of the annulus
        """
        assert 0 <= ring_id < self.ring_num
        return self.ring_radius[ring_id], self.ring_radius[ring_id + 1]

    # ------------------------------------------------------------------
    # Sampling inside a primitive (continuous space)
    # ------------------------------------------------------------------

    def sample_point(
        self,
        ring_id,
        yaw,
        pitch=0.0,
        r_ratio=0.5
    ):
        """
        Sample a 3D point inside a ring primitive using spherical coordinates

        ring_id : which annulus
        yaw     : fixed yaw direction (rad)
        pitch   : vertical angle (rad)
        r_ratio : [0,1] interpolation inside annulus
        """

        r_min, r_max = self.get_annulus_region(ring_id)
        r = r_min + r_ratio * (r_max - r_min)

        x = r * math.cos(pitch) * math.cos(yaw)
        y = r * math.cos(pitch) * math.sin(yaw)
        z = r * math.sin(pitch)

        return torch.tensor(
            [x, y, z],
            dtype=torch.float32,
            device=self.device
        )

    # ------------------------------------------------------------------
    # Rotation matrix (optional, for trajectory alignment)
    # ------------------------------------------------------------------

    def get_rotation_yaw_only(self, yaw):
        """
        Rotation matrix: body -> yaw-aligned frame
        (no pitch / roll)
        """
        c = math.cos(yaw)
        s = math.sin(yaw)

        R = torch.tensor(
            [
                [c, -s, 0.0],
                [s,  c, 0.0],
                [0.0, 0.0, 1.0]
            ],
            dtype=torch.float32,
            device=self.device
        )
        return R

    # ------------------------------------------------------------------
    # Convenience APIs
    # ------------------------------------------------------------------

    def get_all_anchor_points(self, yaw):
        """
        Return all ring anchors along yaw ray
        shape: [ring_num, 3]
        """
        d = self.yaw_ray_direction(yaw).unsqueeze(0)   # [1,3]
        r = self.ring_center.unsqueeze(1)              # [N,1]
        return r * d                                   # [N,3]

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance
