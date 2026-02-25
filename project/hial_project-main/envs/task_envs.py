from __future__ import annotations

import numpy as np
from typing import Optional

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet

from .assets import UR5_URDF, YCB_DIR, assert_assets_exist
from .tasks.ur_robot import UR5
from .tasks.pick_and_place import PickAndPlaceTask


class PnPNewRobotEnv(RobotTaskEnv):
    def __init__(self, render: bool = False, reward_type: str = "modified_sparse", control_type: str = "ee"):
        assert_assets_exist()

        render_mode = "human" if render else "rgb_array"
        sim = PyBullet(render_mode=render_mode, background_color=np.array([150, 222, 246], dtype=np.uint8))
        robot = UR5(
            sim,
            block_gripper=False,
            base_position=np.array([-0.6, 0.0, 0.0], dtype=np.float32),
            control_type=control_type,
            urdf_path=UR5_URDF,
        )
        task = PickAndPlaceTask(sim, reward_type=reward_type, ycb_dir=YCB_DIR)

        super().__init__(robot, task)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        out = super().reset(seed=seed, options=options)

        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = super().step(action)

        if getattr(self.task, "reward_type", None) == "modified_sparse":
            if info.get("is_success", False):
                reward = 1000.0
                terminated = True
            else:
                reward = -1.0

        return obs, float(reward), bool(terminated), bool(truncated), info