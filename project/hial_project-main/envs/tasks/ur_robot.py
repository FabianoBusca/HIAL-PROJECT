from __future__ import annotations

from typing import Optional, Tuple, Union
from pathlib import Path
from collections import namedtuple
import math

import numpy as np
from gymnasium import spaces  # type: ignore

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet


class UR5(PyBulletRobot):
    """UR5 + Robotiq 85 gripper in PyBullet via panda-gym.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool): If True, gripper is fixed (no gripper action dim).
        base_position (np.ndarray): Robot base (x, y, z).
        control_type (str): "ee" (end-effector displacement) or "joints" (delta joint angles).
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
        *,
        urdf_path: Union[str, Path, None] = None,
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3, dtype=np.float32)
        self.block_gripper = bool(block_gripper)
        self.control_type = control_type

        if self.control_type not in ("ee", "joints"):
            raise ValueError(f"control_type must be 'ee' or 'joints', got {self.control_type!r}")

        # UR5 arm is 6-DOF
        self.arm_num_dofs = 6

        n_action = 3 if self.control_type == "ee" else self.arm_num_dofs
        if not self.block_gripper:
            n_action += 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)

        self.arm_joint_indices = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        self.arm_joint_forces = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float32)

        super().__init__(
            sim,
            body_name="ur5",
            file_name=str(urdf_path),
            base_position=base_position,
            action_space=action_space,
            joint_indices=self.arm_joint_indices,
            joint_forces=self.arm_joint_forces,
        )

        self.fingers_indices = np.array([12, 17], dtype=np.int32)

        self.neutral_joint_values = np.array(
            [
                -1.5690622952052096,
                -1.5446774605904932,
                1.343946009733127,
                -1.3708613585093699,
                -1.5707970583733368,
                0.0009377758247187636,
            ],
            dtype=np.float32,
        )

        self.ee_link = 7          # link for position/velocity queries
        self.eef_id = 8           # link index for calculateInverseKinematics

        self._post_init_load()

        self.gripper_range = [0.0, 0.085]
        self.finger_width = float(self.gripper_range[1])

        # Friction settings
        self.sim.set_lateral_friction(self.body_name, int(self.fingers_indices[0]), lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, int(self.fingers_indices[1]), lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, int(self.fingers_indices[0]), spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, int(self.fingers_indices[1]), spinning_friction=0.001)


    def _post_init_load(self) -> None:
        """Parse joints and setup mimic constraints after panda-gym has loaded the body."""
        self.id = self.sim._bodies_idx[self.body_name]

        self.arm_rest_poses = [
            -0.10669020495539147,
            -1.3684361412151338,
            1.6524135640831839,
            -1.854771397805994,
            -1.570735148190762,
            3.0349002737915347,
        ]

        self.__parse_joint_info__()
        self.__post_load__()

    def __parse_joint_info__(self) -> None:
        num_joints = self.sim.physics_client.getNumJoints(self.id)

        jointInfo = namedtuple(
            "jointInfo",
            [
                "id",
                "name",
                "type",
                "damping",
                "friction",
                "lowerLimit",
                "upperLimit",
                "maxForce",
                "maxVelocity",
                "controllable",
            ],
        )

        self.joints = []
        self.controllable_joints = []

        for i in range(num_joints):
            info = self.sim.physics_client.getJointInfo(self.id, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            joint_damping = info[6]
            joint_friction = info[7]
            joint_lower = info[8]
            joint_upper = info[9]
            joint_max_force = info[10]
            joint_max_vel = info[11]

            controllable = (joint_type != self.sim.physics_client.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(joint_id)
                # disable default motors
                self.sim.physics_client.setJointMotorControl2(
                    self.id,
                    joint_id,
                    self.sim.physics_client.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=0,
                )

            self.joints.append(
                jointInfo(
                    joint_id,
                    joint_name,
                    joint_type,
                    joint_damping,
                    joint_friction,
                    joint_lower,
                    joint_upper,
                    joint_max_force,
                    joint_max_vel,
                    controllable,
                )
            )

        if len(self.controllable_joints) < self.arm_num_dofs:
            raise RuntimeError(f"Expected at least {self.arm_num_dofs} controllable joints, got {len(self.controllable_joints)}")

        self.arm_controllable_joints = self.controllable_joints[: self.arm_num_dofs]

        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][: self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][: self.arm_num_dofs]
        self.arm_joint_ranges = [
            (j.upperLimit - j.lowerLimit) for j in self.joints if j.controllable
        ][: self.arm_num_dofs]

    def __post_load__(self) -> None:
        mimic_parent_name = "finger_joint"
        mimic_children_names = {
            "right_outer_knuckle_joint": 1,
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name: str, mimic_children_names: dict) -> None:
        parent_ids = [joint.id for joint in self.joints if joint.name == mimic_parent_name]
        if not parent_ids:
            raise RuntimeError(f"Could not find mimic parent joint named {mimic_parent_name!r} in URDF joint list.")

        self.mimic_parent_id = parent_ids[0]
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name]
            for joint in self.joints
            if joint.name in mimic_children_names
        }

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = self.sim.physics_client.createConstraint(
                self.id,
                self.mimic_parent_id,
                self.id,
                joint_id,
                jointType=self.sim.physics_client.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            self.sim.physics_client.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)


    def open_gripper(self) -> None:
        self.move_gripper(self.gripper_range[1])
        self.finger_width = float(self.gripper_range[1])

    def close_gripper(self) -> None:
        self.move_gripper(self.gripper_range[0])
        self.finger_width = float(self.gripper_range[0])

    def move_gripper(self, open_length: float) -> None:
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        self.sim.physics_client.setJointMotorControl2(
            self.id,
            self.mimic_parent_id,
            self.sim.physics_client.POSITION_CONTROL,
            targetPosition=open_angle,
            force=self.joints[self.mimic_parent_id].maxForce,
            maxVelocity=self.joints[self.mimic_parent_id].maxVelocity,
        )

    def set_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[: self.arm_num_dofs]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        for i, joint_id in enumerate(self.arm_controllable_joints):
            self.sim.physics_client.setJointMotorControl2(
                self.id,
                joint_id,
                self.sim.physics_client.POSITION_CONTROL,
                targetPosition=float(target_arm_angles[i]),
                force=self.joints[joint_id].maxForce,
                maxVelocity=self.joints[joint_id].maxVelocity,
            )

        if not self.block_gripper:
            fingers_ctrl = float(action[-1])
            if fingers_ctrl > 0:
                self.close_gripper()
            else:
                self.open_gripper()

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        ee_displacement = np.asarray(ee_displacement, dtype=np.float32)[:3] * 0.05

        if not np.all(np.isfinite(ee_displacement)):
            return np.array(
                [self.get_joint_angle(joint=idx) for idx in range(self.arm_num_dofs)],
                dtype=np.float32,
            )

        ee_position = np.asarray(self.get_ee_position(), dtype=np.float32)
        target_ee_position = ee_position + ee_displacement

        target_ee_position[2] = float(np.clip(target_ee_position[2], 0.0, 1.0))
        target_ee_position[0] = float(np.clip(target_ee_position[0], -1.0, 1.0))
        target_ee_position[1] = float(np.clip(target_ee_position[1], -1.0, 1.0))

        orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        try:
            joint_poses = self.sim.physics_client.calculateInverseKinematics(
                self.id,
                self.eef_id,
                target_ee_position,
                orientation,
                self.arm_lower_limits,
                self.arm_upper_limits,
                self.arm_joint_ranges,
                self.arm_rest_poses,
                maxNumIterations=50,
            )
            joint_poses = np.asarray(joint_poses[: self.arm_num_dofs], dtype=np.float32)

            if not np.all(np.isfinite(joint_poses)):
                raise ValueError("IK returned non-finite joint poses")

            return joint_poses

        except Exception:
            return np.array(
                [self.get_joint_angle(joint=idx) for idx in range(self.arm_num_dofs)],
                dtype=np.float32,
            )

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute 6 target arm angles from delta joint controls."""
        arm_joint_ctrl = np.asarray(arm_joint_ctrl, dtype=np.float32)[: self.arm_num_dofs] * 0.05

        current_arm_joint_angles = np.array(
            [self.get_joint_angle(joint=idx) for idx in range(self.arm_num_dofs)],
            dtype=np.float32,
        )
        return current_arm_joint_angles + arm_joint_ctrl

    def get_obs(self) -> np.ndarray:
        ee_position = np.asarray(self.get_ee_position(), dtype=np.float32)
        ee_velocity = np.asarray(self.get_ee_velocity(), dtype=np.float32)

        if not self.block_gripper:
            obs = np.concatenate((ee_position, ee_velocity, np.array([self.get_fingers_width()], dtype=np.float32)))
        else:
            obs = np.concatenate((ee_position, ee_velocity))

        return obs.astype(np.float32, copy=False)

    def reset(self) -> None:
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self) -> None:
        """Reset arm joints to rest poses."""
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            self.sim.physics_client.resetJointState(self.id, joint_id, float(rest_pose))

    def reset_gripper(self) -> None:
        if self.block_gripper:
            self.close_gripper()
        else:
            self.open_gripper()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        return float(self.finger_width)

    def get_ee_position(self) -> np.ndarray:
        """End-effector position (x, y, z)."""
        return np.asarray(self.get_link_position(self.ee_link), dtype=np.float32)

    def get_ee_velocity(self) -> np.ndarray:
        """End-effector velocity (vx, vy, vz)."""
        return np.asarray(self.get_link_velocity(self.ee_link), dtype=np.float32)