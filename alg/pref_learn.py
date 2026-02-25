from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import aprel.utils.util_functions as util_funs
from alg.banana import feature_function
from aprel.basics.environment import Environment
from aprel.basics.trajectory import TrajectorySet
from aprel.learning.belief_models import SamplingBasedBelief
from aprel.learning.data_types import Preference, PreferenceQuery
from aprel.learning.user_models import SoftmaxUser
from aprel.querying.query_optimizer import QueryOptimizerDiscreteTrajectorySet
from envs.task_envs import PnPNewRobotEnv
from utils.env_wrappers import (
    ActionNormalizer,
    AprelGymAdapter,
    ResetWrapper,
    TimeLimitWrapper,
    TrajectoryRecord,
)


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
    pass


def learn_weights(
    traj_set: TrajectorySet,
    *,
    num_queries: int = 10,
    seed: int = 0,
    acquisition_function: str
) -> np.ndarray:
    """Run an active preference learning loop to infer reward feature weights.

    Uses APReL's query optimizer together with a SamplingBasedBelief.  At each
    iteration a pair of trajectories is selected and shown to the human annotator
    via query.visualize().  The annotator's response updates the belief.

    Args:
        traj_set: The discrete set of candidate trajectories to query over.
        num_queries: Number of preference queries to collect from the human.
        seed: Seed for numpy's global RNG.
        acquisition_function: Name of an acquisition function to use:
            - `disagreement`: Based on `Katz. et al. (2019) <https://arxiv.org/abs/1907.05575>`_.
            - `mutual_information`: Based on `Bıyık et al. (2019) <https://arxiv.org/abs/1910.04365>`_.
            - `random`: Randomly chooses a query.
            - `regret`: Based on `Wilde et al. (2020) <https://arxiv.org/abs/2005.04067>`_.
            - `thompson`: Based on `Tucker et al. (2019) <https://arxiv.org/abs/1909.12316>`_.
            - `volume_removal`: Based on `Sadigh et al. (2017) <http://m.roboticsproceedings.org/rss13/p53.pdf>`_ and `Bıyık et al. <https://arxiv.org/abs/1904.02209>`_.

    Returns:
        A float32 array of shape (feature_dim,) containing the posterior
        mean reward weights after all queries have been collected.
    """
    pass


def save_weights(weights: np.ndarray, out_path: Path) -> None:
    """Serialise learned feature weights to a two-column CSV file.

    Args:
        weights: 1-D array of reward feature weights.
        out_path: Destination file path (will be created or overwritten).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_index", "weight"])
        for i, val in enumerate(weights.tolist()):
            writer.writerow([i, float(val)])


def main() -> None:
    """load trajectory records, run preference learning, and save weights.

    1. Load TrajectoryRecord objects from
       ``<repo_root>/saved/trajectory_records.json`` (produced by banana.py).
    2. Convert records into an APReL TrajectorySet.
    3. Run learn_weights to interactively collect human preferences and
       infer posterior mean reward weights.
    4. Save the resulting weights to ``<repo_root>/saved/feature_weights.csv``.

    Raises:
        FileNotFoundError: If ``trajectory_records.json`` does not exist.
    """
    repo_root = Path(__file__).resolve().parents[1]
    saved_dir = repo_root / "saved"

    records_path = saved_dir / "trajectory_records.json"
    if not records_path.exists():
        raise FileNotFoundError(
            f"Missing trajectory records at {records_path}. "
            "Run banana.py first to generate clips and records."
        )

    env = setup_environment(render=False)
    aprel_env = Environment(AprelGymAdapter(env), feature_func=feature_function)

    records = json.loads(records_path.read_text())
    trajectories = TrajectorySet(
        [TrajectoryRecord.from_json(r).to_aprel(aprel_env) for r in records]
    )

    out_path = saved_dir / "feature_weights.csv"

    try:
        weights = learn_weights(trajectories, num_queries=10, seed=0)
    except Exception as e:
        raise
    finally:
        env.close()

    save_weights(weights, out_path)
    print(f"\nSaved learned weights to: {out_path}")


if __name__ == "__main__":
    main()