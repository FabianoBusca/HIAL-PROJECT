from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


def prepare_demo_pool(
    demo_dir: Path,
    *,
    delimiter: str = " ",
    marker_value: float = np.inf,
    marker_column: int = 0,
    reshape_reward_done: bool = True,
    verbose: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """
    Load demonstrations from CSV files in `demo_dir`.

    Episodes are separated by a marker row where state_traj[i][marker_column] == marker_value.
    """
    demo_dir = Path(demo_dir)

    state_traj = np.genfromtxt(demo_dir / "state_traj.csv", delimiter=delimiter)
    action_traj = np.genfromtxt(demo_dir / "action_traj.csv", delimiter=delimiter)
    next_state_traj = np.genfromtxt(demo_dir / "next_state_traj.csv", delimiter=delimiter)
    reward_traj = np.genfromtxt(demo_dir / "reward_traj.csv", delimiter=delimiter)
    done_traj = np.genfromtxt(demo_dir / "done_traj.csv", delimiter=delimiter)

    if reshape_reward_done:
        reward_traj = np.reshape(reward_traj, (-1, 1))
        done_traj = np.reshape(done_traj, (-1, 1))

    if verbose:
        print(f"[prepare_demo_pool] reward traj shape: {reward_traj.shape}")
        print(f"[prepare_demo_pool] done traj shape:   {done_traj.shape}")

    starting_ids = [i for i in range(state_traj.shape[0]) if state_traj[i][marker_column] == marker_value]
    total = len(starting_ids)
    if total == 0:
        raise ValueError(
            f"No episode markers found (state_traj[:, {marker_column}] == {marker_value}). "
            f"Check delimiter/marker settings or file contents."
        )

    demos: List[Dict[str, np.ndarray]] = []
    for i in range(total):
        start = starting_ids[i]
        end = starting_ids[i + 1] if i < total - 1 else state_traj.shape[0]
        sl = slice(start + 1, end)

        demos.append(
            {
                "state_trajectory": state_traj[sl, :].copy(),
                "action_trajectory": action_traj[sl, :].copy(),
                "next_state_trajectory": next_state_traj[sl, :].copy(),
                "reward_trajectory": reward_traj[sl, :].copy(),
                "done_trajectory": done_traj[sl, :].copy(),
            }
        )

    return demos