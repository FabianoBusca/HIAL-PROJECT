from __future__ import annotations
from pathlib import Path

# Absolute path to .../hial_project-main/envs
ENVS_DIR = Path(__file__).resolve().parent

TASKS_DIR = ENVS_DIR / "tasks"
UR5_URDF = TASKS_DIR / "ur5" / "urdf" / "ur5_robotiq_85.urdf"
YCB_DIR = TASKS_DIR / "ycb_objects"

def assert_assets_exist() -> None:
    if not UR5_URDF.is_file():
        raise FileNotFoundError(f"UR5 URDF not found: {UR5_URDF}")
    if not YCB_DIR.is_dir():
        raise FileNotFoundError(f"YCB dir not found: {YCB_DIR}")