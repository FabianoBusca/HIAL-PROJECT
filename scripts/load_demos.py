from __future__ import annotations

from pathlib import Path
from time import sleep

import numpy as np

from envs.task_envs import PnPNewRobotEnv
from utils.env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper
from utils.demos import prepare_demo_pool


def main() -> None:
    env = PnPNewRobotEnv(render=True)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)

    repo_root = Path(__file__).resolve().parents[1]
    demo_dir = repo_root / "demo_data" / "PickAndPlace"

    demos = prepare_demo_pool(demo_dir, verbose=True)

    for i, demo in enumerate(demos, start=1):
        state_traj = demo["state_trajectory"]
        action_traj = demo["action_trajectory"]
        done_traj = np.asarray(demo["done_trajectory"]).reshape(-1)

        if state_traj.ndim != 2:
            raise ValueError(f"[Demo {i}] state_trajectory must be 2D, got shape {state_traj.shape}")
        if action_traj.ndim != 2:
            raise ValueError(f"[Demo {i}] action_trajectory must be 2D, got shape {action_traj.shape}")
        if state_traj.shape[1] < 10:
            raise ValueError(
                f"[Demo {i}] state_trajectory has {state_traj.shape[1]} dims, expected >= 10 to slice [7:10] for object_pos"
            )

        T = min(action_traj.shape[0], done_traj.shape[0])
        if T <= 0:
            print(f"[Demo {i}] empty trajectory; skipping.")
            continue

        object_pos = np.asarray(state_traj[0][7:10], dtype=np.float32)
        if not np.all(np.isfinite(object_pos)):
            raise ValueError(f"[Demo {i}] initial object_pos is non-finite: {object_pos}")

        obs, info = env.reset(
            seed=0,
            options={"whether_random": False, "object_pos": object_pos},
        )

        step = 0
        while True:
            if step >= T:
                break

            action = np.asarray(action_traj[step], dtype=np.float32)

            if not np.all(np.isfinite(action)):
                print(f"[Demo {i}] step={step}: non-finite action {action}; stopping replay.")
                break

            obs, reward, terminated, truncated, info = env.step(action)

            env.render()
            sleep(0.01)

            demo_done = bool(done_traj[step] != 0)
            if demo_done:
                break
            if terminated or truncated:
                break

            step += 1

        print(f"[Demo {i}]: env_success={info.get('is_success', False)}  steps={step+1}")
        print("***********************************")

    env.close()
    print("All finished")


if __name__ == "__main__":
    main()