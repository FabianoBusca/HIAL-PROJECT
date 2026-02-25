from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from d3rlpy.algos import AWACConfig
from d3rlpy.dataset import MDPDataset

from alg.banana import feature_function
from envs.task_envs import PnPNewRobotEnv
from utils.demos import prepare_demo_pool
from utils.env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper, reconstruct_state

@dataclass
class EpisodeTransitions:
    """Vectorised transition data for a single episode.

    Attributes:
        observations: (T, obs_dim) float32
        actions: (T, act_dim) float32
        rewards: (T,) float32
        next_observations: (T, obs_dim) float32
        terminals: (T,) float32, 1.0 if terminated else 0.0
        success: True if episode succeeded at any time
        T: number of transitions in the episode
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    success: bool
    T: int

def setup_environment(*, render: bool = False) -> Any:
    """Construct and initialise the Pick-and-Place environment with standard wrappers.

    Wrapper stack (inner → outer):
    PnPNewRobotEnv → ResetWrapper → ActionNormalizer → TimeLimitWrapper (150 steps).

    The environment is seeded with seed=0 immediately after construction.

    Args:
        render: If True, opens a PyBullet GUI window.

    Returns:
        The fully wrapped, reset environment.
    """
    env = PnPNewRobotEnv(render=render)
    env = ResetWrapper(env)
    env = ActionNormalizer(env)
    env = TimeLimitWrapper(env, max_steps=150)
    env.reset(seed=0)
    return env


def load_weights(path: Path) -> np.ndarray:
    """Load learned reward weights from a two-column CSV file.

    The file is expected to have columns:
        - feature_index
        - weight

    Args:
        path: Path to saved weights.

    Returns:
        weights: float32 array of shape (feature_dim,).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no weights were parsed.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing learned weights at {path}. Run preference learning first (pref_learn.py)."
        )

    weights: List[float] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            weights.append(float(row["weight"]))

    if not weights:
        raise ValueError(f"No weights found in {path}")

    return np.asarray(weights, dtype=np.float32)


def episode_return(
    traj_pairs: List[Tuple[Dict[str, np.ndarray], np.ndarray]],
    weights: np.ndarray,
) -> float:
    """Compute trajectory-level return: R(τ) = w^T φ(τ).

    Args:
        traj_pairs: Episode as a list of (state_dict, action) pairs.
        weights: Learned weight vector w.

    Returns:
        Scalar episode return.
    """
    phi = feature_function(traj_pairs)
    if phi.shape[0] != weights.shape[0]:
        raise ValueError(f"Feature dim mismatch: phi={phi.shape}, w={weights.shape}")
    return float(np.dot(weights, phi))


def distribute_return(R: float, T: int) -> np.ndarray:
    """Convert episode return into per-step rewards by uniform distribution.

    Args:
        R: Scalar episode return.
        T: Episode length.

    Returns:
        rewards: float32 array of shape (T,) where each element equals R / T.
    """
    if T <= 0:
        return np.zeros((0,), dtype=np.float32)
    return np.full((T,), R / float(T), dtype=np.float32)


def rollout_expert(
    env: Any,
    action_seq: np.ndarray,
    *,
    options: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    stored_rewards: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[Dict[str, np.ndarray], np.ndarray]], EpisodeTransitions]:
    """Replay an expert action sequence to collect transitions.

    Args:
        env: Wrapped environment.
        action_seq: (T, act_dim) expert actions in normalized range [-1, 1].
        options: Reset options (e.g., fixed object_pos).
        seed: Reset seed.
        stored_rewards: 1-D float32 array of per-step rewards from the demo
            CSV.  If provided, success is set to True iff any stored reward
            exceeds SUCCESS_REWARD_THRESHOLD.  Falls back to live
            info["is_success"] if None.

    Returns:
        traj_pairs: (state_dict, action) pairs for learned reward computation.
        ep: EpisodeTransitions with rewards initialised to zero (fill later).
    """
    traj_pairs: List[Tuple[Dict[str, np.ndarray], np.ndarray]] = []

    obs_dict, _info = env.reset(seed=seed, options=options)
    obs = reconstruct_state(obs_dict)

    obs_buf: List[np.ndarray] = []
    act_buf: List[np.ndarray] = []
    next_obs_buf: List[np.ndarray] = []
    terminals_buf: List[float] = []

    live_success = False

    for t in range(min(int(action_seq.shape[0]), 150)):
        action = np.asarray(action_seq[t], dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        next_obs_dict, _r_env, terminated, truncated, info = env.step(action)

        traj_pairs.append((obs_dict, action))

        next_obs = reconstruct_state(next_obs_dict)

        obs_buf.append(obs)
        act_buf.append(action)
        next_obs_buf.append(next_obs)
        terminals_buf.append(1.0 if bool(terminated) else 0.0)

        live_success = live_success or bool(info.get("is_success", False))

        obs_dict = next_obs_dict
        obs = next_obs

        if terminated or truncated:
            break

    # Prefer stored reward signal over live env signal to avoid replay divergence
    if stored_rewards is not None:
        success = bool(np.any(stored_rewards >= 900))
    else:
        success = live_success

    T = len(obs_buf)
    if T == 0:
        return traj_pairs, EpisodeTransitions(
            observations=np.zeros((0, 1), dtype=np.float32),
            actions=np.zeros((0, 1), dtype=np.float32),
            rewards=np.zeros((0,), dtype=np.float32),
            next_observations=np.zeros((0, 1), dtype=np.float32),
            terminals=np.zeros((0,), dtype=np.float32),
            success=success,
            T=0,
        )

    obs_arr = np.stack(obs_buf).astype(np.float32, copy=False)
    act_arr = np.stack(act_buf).astype(np.float32, copy=False)
    next_obs_arr = np.stack(next_obs_buf).astype(np.float32, copy=False)
    terminals_arr = np.asarray(terminals_buf, dtype=np.float32)
    rewards_arr = np.zeros((T,), dtype=np.float32)  # filled after computing learned reward

    return traj_pairs, EpisodeTransitions(
        observations=obs_arr,
        actions=act_arr,
        rewards=rewards_arr,
        next_observations=next_obs_arr,
        terminals=terminals_arr,
        success=success,
        T=T,
    )


def rollout_policy(
    env: Any,
    algo: Any,
    *,
    seed: int,
) -> Tuple[List[Tuple[Dict[str, np.ndarray], np.ndarray]], EpisodeTransitions]:
    """Roll out the current policy for one episode.

    Args:
        env: Wrapped environment.
        algo: d3rlpy algorithm (AWAC).
        seed: Reset seed.

    Returns:
        traj_pairs: (state_dict, action) pairs for learned reward computation.
        ep: EpisodeTransitions with rewards initialised to zero (fill later).
    """
    traj_pairs: List[Tuple[Dict[str, np.ndarray], np.ndarray]] = []

    obs_dict, _info = env.reset(seed=seed)
    obs = reconstruct_state(obs_dict)

    obs_buf: List[np.ndarray] = []
    act_buf: List[np.ndarray] = []
    next_obs_buf: List[np.ndarray] = []
    terminals_buf: List[float] = []

    success = False

    for _t in range(150):
        # d3rlpy expects (batch, obs_dim)
        action = algo.predict(obs[None, :])[0].astype(np.float32, copy=False)
        action = np.clip(action, -1.0, 1.0)

        next_obs_dict, _r_env, terminated, truncated, info = env.step(action)

        traj_pairs.append((obs_dict, action))

        next_obs = reconstruct_state(next_obs_dict)

        obs_buf.append(obs)
        act_buf.append(action)
        next_obs_buf.append(next_obs)
        terminals_buf.append(1.0 if bool(terminated) else 0.0)

        success = success or bool(info.get("is_success", False))

        obs_dict = next_obs_dict
        obs = next_obs

        if terminated or truncated:
            break

    T = len(obs_buf)
    if T == 0:
        return traj_pairs, EpisodeTransitions(
            observations=np.zeros((0, 1), dtype=np.float32),
            actions=np.zeros((0, 1), dtype=np.float32),
            rewards=np.zeros((0,), dtype=np.float32),
            next_observations=np.zeros((0, 1), dtype=np.float32),
            terminals=np.zeros((0,), dtype=np.float32),
            success=success,
            T=0,
        )

    obs_arr = np.stack(obs_buf).astype(np.float32, copy=False)
    act_arr = np.stack(act_buf).astype(np.float32, copy=False)
    next_obs_arr = np.stack(next_obs_buf).astype(np.float32, copy=False)
    terminals_arr = np.asarray(terminals_buf, dtype=np.float32)
    rewards_arr = np.zeros((T,), dtype=np.float32)  # filled later

    return traj_pairs, EpisodeTransitions(
        observations=obs_arr,
        actions=act_arr,
        rewards=rewards_arr,
        next_observations=next_obs_arr,
        terminals=terminals_arr,
        success=success,
        T=T,
    )

def build_dataset(episodes: List[EpisodeTransitions]) -> MDPDataset:
    """Build an MDPDataset by concatenating transitions from all episodes.

    Args:
        episodes: List of EpisodeTransitions to include in the dataset.

    Returns:
        MDPDataset ready for d3rlpy training.
    """
    valid = [ep for ep in episodes if ep.T > 0]
    if not valid:
        return MDPDataset(
            observations=np.zeros((0, 1), dtype=np.float32),
            actions=np.zeros((0, 1), dtype=np.float32),
            rewards=np.zeros((0,), dtype=np.float32),
            terminals=np.zeros((0,), dtype=np.float32),
        )

    obs = np.concatenate([ep.observations for ep in valid], axis=0)
    actions = np.concatenate([ep.actions for ep in valid], axis=0)
    rewards = np.concatenate([ep.rewards for ep in valid], axis=0)
    terminals = np.concatenate([ep.terminals for ep in valid], axis=0)

    return MDPDataset(
        observations=obs,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )


def evaluate_success_rate(env: Any, algo: Any, *, runs: int = 10, seed: int = 0) -> float:
    """Evaluate success rate over runs episodes.

    Args:
        env: Wrapped environment.
        algo: Trained d3rlpy algorithm.
        runs: Number of evaluation episodes.
        seed: Base seed.

    Returns:
        Fraction of episodes where success was detected, in [0.0, 1.0].
    """
    successes = 0
    for k in range(int(runs)):
        _traj_pairs, ep = rollout_policy(env, algo, seed=seed + k + 10_000_000)
        successes += 1 if ep.success else 0
    return float(successes) / float(runs)


def save_curve_csv(xs: List[int], ys: List[float], out_path: Path) -> None:
    """Save learning curve as CSV with columns (env_steps, avg_success_rate).

    Args:
        xs: Environment step counts.
        ys: Corresponding average success rates.
        out_path: Destination CSV path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env_steps", "avg_success_rate"])
        for x, y in zip(xs, ys):
            w.writerow([int(x), float(y)])


def plot_curve(xs: List[int], ys: List[float], out_path: Path) -> None:
    """Plot learning curve and save as PNG.

    Args:
        xs: Environment step counts.
        ys: Corresponding average success rates.
        out_path: Destination PNG path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Environment steps")
    plt.ylabel("Average success rate (10 test runs)")
    plt.title("RLfD policy learning curve (AWAC)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    """Train AWAC from demonstrations using a learned reward function."""
    repo_root = Path(__file__).resolve().parents[1]
    saved_dir = repo_root / "saved"
    models_dir = saved_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    weights = load_weights(saved_dir / "feature_weights.csv")
    print(f"Loaded learned weights (dim={weights.shape[0]}): {np.round(weights, 3)}")

    env = setup_environment(render=False)

    demo_dir = repo_root / "demo_data" / "PickAndPlace"
    demos = prepare_demo_pool(demo_dir, verbose=True)
    print(f"\nLoaded {len(demos)} expert demos from: {demo_dir}")

    episodes: List[EpisodeTransitions] = []

    for i, demo in enumerate(demos):
        state_traj = np.asarray(demo["state_trajectory"])
        action_traj = np.asarray(demo["action_trajectory"], dtype=np.float32)

        reward_traj = np.asarray(demo["reward_trajectory"], dtype=np.float32).flatten()

        init_object_pos = np.asarray(state_traj[0][7:10], dtype=np.float32)
        options: Dict[str, Any] = {"whether_random": False, "object_pos": init_object_pos}

        traj_pairs, ep = rollout_expert(
            env,
            action_traj,
            options=options,
            seed=i,
            stored_rewards=reward_traj,
        )

        R = episode_return(traj_pairs, weights)
        ep.rewards = distribute_return(R, ep.T)

        episodes.append(ep)
        print(f"[demo {i:02d}] T={ep.T} success={ep.success} R={R:.3f}")

    dataset = build_dataset(episodes)
    print(f"\nInitial dataset size: {dataset.size()} transitions")


    algo = AWACConfig().create(device="cpu")

    print("\nWarm-start")
    algo.fit(dataset, n_steps=20_000, n_steps_per_epoch=1_000)

    max_env_steps = 10_000
    save_every = 1_000
    eval_every = 1_000

    env_steps = int(sum(ep.T for ep in episodes))
    next_save = ((env_steps // save_every) + 1) * save_every
    next_eval = ((env_steps // eval_every) + 1) * eval_every

    curve_x: List[int] = []
    curve_y: List[float] = []

    print("\nOnline RLfD training")
    while env_steps < max_env_steps:
        traj_pairs, ep = rollout_policy(env, algo, seed=env_steps + 123)

        if ep.T == 0:
            print("[warning] got empty episode (T=0). skipping.")
            continue

        R = episode_return(traj_pairs, weights)
        ep.rewards = distribute_return(R, ep.T)

        episodes.append(ep)
        env_steps += ep.T

        dataset = build_dataset(episodes)

        update_steps = 1_000
        algo.fit(dataset, n_steps=update_steps, n_steps_per_epoch=update_steps)

        while env_steps >= next_save:
            ckpt = models_dir / f"awac_step_{next_save:06d}.pt"
            algo.save(str(ckpt))
            print(f"[save] {ckpt}")
            next_save += save_every

        while env_steps >= next_eval:
            sr = evaluate_success_rate(env, algo, runs=10, seed=next_eval)
            curve_x.append(next_eval)
            curve_y.append(sr)
            print(f"[eval] steps={next_eval} avg_success_rate={sr:.3f}")

            save_curve_csv(curve_x, curve_y, saved_dir / "learning_curve.csv")
            plot_curve(curve_x, curve_y, saved_dir / "learning_curve.png")

            next_eval += eval_every

    final_ckpt = models_dir / f"awac_final_{env_steps:06d}.pt"
    algo.save(str(final_ckpt))
    print(f"\nFinal model saved to: {final_ckpt}")

    save_curve_csv(curve_x, curve_y, saved_dir / "learning_curve.csv")
    plot_curve(curve_x, curve_y, saved_dir / "learning_curve.png")

    env.close()

if __name__ == "__main__":
    main()