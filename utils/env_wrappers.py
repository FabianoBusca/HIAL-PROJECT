from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple, Literal

import gymnasium as gym
import numpy as np

from aprel.basics.trajectory import Trajectory as AprelTrajectory
from aprel.basics.environment import Environment as AprelEnvironment


class ActionNormalizer(gym.ActionWrapper):
    """
    Normalize actions from [-1, 1] to [low, high] and vice versa.
    """

    def action(self, action: np.ndarray) -> np.ndarray:
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2.0
        reloc_factor = low + scale_factor  # center

        action = np.asarray(action, dtype=np.float32)
        action = action * scale_factor + reloc_factor
        return np.clip(action, low, high)

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2.0
        reloc_factor = low + scale_factor

        action = np.asarray(action, dtype=np.float32)
        action = (action - reloc_factor) / scale_factor
        return np.clip(action, -1.0, 1.0)


class TimeLimitWrapper(gym.Wrapper):
    """
    Time limit wrapper for Gymnasium API.

    Sets truncated=True once max_steps is reached.
    """

    def __init__(self, env: gym.Env, max_steps: int = 100):
        super().__init__(env)
        self.max_steps = int(max_steps)
        self.current_step = 0

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
            **kwargs,
    ):
        self.current_step = 0

        return self.env.reset(seed=seed, options=options, **kwargs)

    def step(self, action):
        self.current_step += 1

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.current_step >= self.max_steps and not terminated:
            truncated = True
            info = dict(info)
            info["TimeLimit.truncated"] = True

        return obs, reward, terminated, truncated, info


class ResetWrapper(gym.Wrapper):
    """
    Adds custom reset behavior:
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
            **kwargs,
    ):
        options = dict(options or {})

        whether_random = bool(options.get("whether_random", True))
        object_pos = options.get("object_pos", None)

        obs, info = self.env.reset(seed=seed, options=options)

        if whether_random:
            return obs, info

        if object_pos is None:
            raise ValueError("ResetWrapper: options['object_pos'] must be provided when whether_random=False")

        object_pos = np.asarray(object_pos, dtype=np.float32)

        with self.env.sim.no_rendering():
            self.env.task.sim.set_base_pose("target", self.env.task.goal, np.array([0, 0, 0, 1], dtype=np.float32))
            self.env.task.sim.set_base_pose(
                "object",
                object_pos,
                np.array([0.0, 0.0, 0.7071, 0.7071], dtype=np.float32),
            )

        robot_obs = np.asarray(self.env.robot.get_obs(), dtype=np.float32)
        task_obs = np.asarray(self.env.task.get_obs(), dtype=np.float32)
        observation = np.concatenate([robot_obs, task_obs]).astype(np.float32, copy=False)

        achieved_goal = np.asarray(self.env.task.get_achieved_goal(), dtype=np.float32)
        desired_goal = np.asarray(self.env.task.get_goal(), dtype=np.float32)

        obs = {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if info.get("is_success", False):
            terminated = True

        return obs, reward, terminated, truncated, info


def reconstruct_state(state: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Flatten (observation + desired_goal) for algorithms that want a single vector.
    """
    obs = np.asarray(state["observation"], dtype=np.float32)
    goal = np.asarray(state["desired_goal"], dtype=np.float32)
    return np.concatenate((obs, goal)).astype(np.float32, copy=False)


class AprelGymAdapter:
    """
    Adapter for Gym-based APReL API:
    """

    def __init__(self, env: gym.Env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = getattr(env, "observation_space", None)
        self._seed: Optional[int] = None

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def seed(self, seed: int) -> None:
        self._seed = int(seed)

    def reset(self) -> Any:
        obs, _info = self.env.reset(seed=self._seed)
        return obs

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        info = dict(info)
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        return obs, float(reward), done, info

    def close(self) -> None:
        self.env.close()


@dataclass
class TrajectoryRecord:
    """
    Trajectory wrapper to serialize Aprel trajectories.
      - features: used by APReL optimizer/belief updates
      - clip_path: used to query the human
    """
    clip_path: str
    features: np.ndarray  # shape (d,)

    def to_json(self) -> dict:
        return {
            "clip_path": self.clip_path,
            "features": self.features.astype(float).tolist(),
        }

    def to_aprel(self, aprel_env: AprelEnvironment) -> AprelTrajectory:
        """
        Create an APReL Trajectory with empty rollout and attached precomputed features.
        """
        t = AprelTrajectory(aprel_env, [], clip_path=self.clip_path)
        t.features = np.asarray(self.features, dtype=np.float32)
        return t

    @staticmethod
    def from_json(d: dict) -> "TrajectoryRecord":
        return TrajectoryRecord(
            clip_path=str(d["clip_path"]),
            features=np.asarray(d["features"], dtype=np.float32),
        )