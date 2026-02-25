from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import d3rlpy
import numpy as np

from envs.task_envs import PnPNewRobotEnv
from utils.env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper, reconstruct_state


def setup_environment(*, render: bool = False) -> Any:
    env = PnPNewRobotEnv(render=render)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)
    env.reset(seed=0)
    return env

def rollout_policy(
    env: Any,
    algo: Any,
    *,
    seed: int,
    render: bool = False,
) -> Tuple[bool, int, Dict[str, Any]]:
    """Run one episode. Returns (success, T, last_info)."""
    obs_dict, _ = env.reset(seed=seed)
    obs = reconstruct_state(obs_dict)

    success = False
    last_info: Dict[str, Any] = {}

    for t in range(150):
        action = algo.predict(obs[None, :])[0].astype(np.float32, copy=False)
        action = np.clip(action, -1.0, 1.0)

        obs_dict, _r, terminated, truncated, info = env.step(action)
        last_info = dict(info)
        success = success or bool(info.get("is_success", False))

        obs = reconstruct_state(obs_dict)

        if render:
            env.render()

        if terminated or truncated:
            return success, (t + 1), last_info

    return success, 150, last_info


def main() -> None:

    repo_root = Path(__file__).resolve().parents[1]

    step = int(400000)
    ckpt = repo_root / "saved" / "models" / f"awac_step_{step:06d}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Cannot render: missing {ckpt}")

    gui_env = setup_environment(render=True)
    algo =  d3rlpy.load_learnable(str(ckpt), device="cpu")

    print(f"\n=== Rendering one deterministic rollout for {ckpt.name} ===")
    succ, T, info = rollout_policy(gui_env, algo, seed=0, render=True)
    print(f"render result: success={succ}  T={T}  info.is_success={info.get('is_success', None)}")

    gui_env.close()


if __name__ == "__main__":
    main()