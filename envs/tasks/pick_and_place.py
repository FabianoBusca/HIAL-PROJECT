from __future__ import annotations

from typing import Any, Dict, Union, Tuple
from pathlib import Path

import numpy as np
from gymnasium.utils import seeding

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class PickAndPlaceTask(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.15,
        goal_xy_range: Tuple[float, float] = (0.0, 0.0),
        obj_xy_range: Tuple[float, float] = (0.1, 0.1),
        *,
        object_urdf: str = "011_banana.urdf",
        target_urdf: str = "029_plate.urdf",
        ycb_dir: Union[str, Path, None] = None,
        debug_draw_areas: bool = True,
        object_global_scaling: float = 0.06,
        target_global_scaling: float = 0.08,
        target_fixed_base: bool = True,
    ) -> None:
        super().__init__(sim)

        self.np_random, _ = seeding.np_random(None)

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold

        # Range centers
        self.goal_range_center = np.array([0.0, -0.2, 0.02], dtype=np.float32)
        self.object_range_center = np.array([-0.3, 0.0, 0.02], dtype=np.float32)
        goal_xy_range = np.asarray(goal_xy_range, dtype=np.float32)
        obj_xy_range = np.asarray(obj_xy_range, dtype=np.float32)

        # Sampling bounds
        self.goal_range_low = np.array(
            [
                self.goal_range_center[0] - goal_xy_range[0] / 2.0,
                self.goal_range_center[1] - goal_xy_range[1] / 2.0,
                self.goal_range_center[2],
            ],
            dtype=np.float32,
        )
        self.goal_range_high = np.array(
            [
                self.goal_range_center[0] + goal_xy_range[0] / 2.0,
                self.goal_range_center[1] + goal_xy_range[1] / 2.0,
                self.goal_range_center[2],
            ],
            dtype=np.float32,
        )

        self.obj_range_low = np.array(
            [
                self.object_range_center[0] - obj_xy_range[0] / 2.0,
                self.object_range_center[1] - obj_xy_range[1] / 2.0,
                self.object_range_center[2],
            ],
            dtype=np.float32,
        )
        self.obj_range_high = np.array(
            [
                self.object_range_center[0] + obj_xy_range[0] / 2.0,
                self.object_range_center[1] + obj_xy_range[1] / 2.0,
                self.object_range_center[2],
            ],
            dtype=np.float32,
        )

        self.object_urdf_path = (ycb_dir / object_urdf).resolve()
        self.target_urdf_path = (ycb_dir / target_urdf).resolve()

        self.debug_draw_areas = debug_draw_areas
        self.object_global_scaling = float(object_global_scaling)
        self.target_global_scaling = float(target_global_scaling)
        self.target_fixed_base = bool(target_fixed_base)

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(
                target_position=np.zeros(3, dtype=np.float32),
                distance=0.9,
                yaw=45,
                pitch=-30,
            )

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)

        self.sim.loadURDF(
            body_name="object",
            fileName=str(self.object_urdf_path),
            basePosition=np.array([0.0, 0.15, 0.02], dtype=np.float32),
            baseOrientation=np.array([0.0, 0.0, 0.7071, 0.7071], dtype=np.float32),
            useFixedBase=False,
            globalScaling=self.object_global_scaling,
        )

        self.sim.loadURDF(
            body_name="target",
            fileName=str(self.target_urdf_path),
            basePosition=np.array([0.0, -0.15, 0.02], dtype=np.float32),
            useFixedBase=self.target_fixed_base,
            globalScaling=self.target_global_scaling,
        )

        if self.debug_draw_areas:
            object_area_points = [
                [-0.25, -0.05, 0.001],
                [-0.35, -0.05, 0.001],
                [-0.35, 0.05, 0.001],
                [-0.25, 0.05, 0.001],
            ]
            pc = self.sim.physics_client
            pc.addUserDebugLine(object_area_points[0], object_area_points[1], lineColorRGB=[1, 0, 0], lineWidth=2)
            pc.addUserDebugLine(object_area_points[1], object_area_points[2], lineColorRGB=[1, 0, 0], lineWidth=2)
            pc.addUserDebugLine(object_area_points[2], object_area_points[3], lineColorRGB=[1, 0, 0], lineWidth=2)
            pc.addUserDebugLine(object_area_points[3], object_area_points[0], lineColorRGB=[1, 0, 0], lineWidth=2)

    def get_obs(self) -> np.ndarray:
        """
        Observation: object position (3), rotation quaternion (4),
        linear velocity (3), angular velocity (3) -> total 13 dims.
        """
        object_position = np.asarray(self.sim.get_base_position("object"), dtype=np.float32)
        object_rotation = np.asarray(self.sim.get_base_rotation("object"), dtype=np.float32)
        object_velocity = np.asarray(self.sim.get_base_velocity("object"), dtype=np.float32)
        object_angular_velocity = np.asarray(self.sim.get_base_angular_velocity("object"), dtype=np.float32)
        observation = np.concatenate(
            [object_position, object_rotation, object_velocity, object_angular_velocity]
        ).astype(np.float32, copy=False)
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        """Achieved goal = object position."""
        return np.asarray(self.sim.get_base_position("object"), dtype=np.float32)

    def reset(self) -> None:
        """
        Reset task: sample goal and object position and apply poses.
        Note: seeding is handled by the environment; we sample with self.np_random.
        """
        self.goal = self._sample_goal()
        object_position = self._sample_object()

        # Target orientation
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

        # Object orientation
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.7071, 0.7071], dtype=np.float32))

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal using the task RNG (reproducible with env.reset(seed=...))."""
        return self.np_random.uniform(self.goal_range_low, self.goal_range_high).astype(np.float32)

    def _sample_object(self) -> np.ndarray:
        """Sample initial object position using the task RNG (reproducible with env.reset(seed=...))."""
        return self.np_random.uniform(self.obj_range_low, self.obj_range_high).astype(np.float32)

    def is_success(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any] = None,
    ) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float32)

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any] = None,
    ) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            # 0 if success, -1 if failure
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        return -np.array(d, dtype=np.float32)