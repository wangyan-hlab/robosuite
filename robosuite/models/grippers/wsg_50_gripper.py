"""
Two-Finger WSG Finray Gripper.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class WSGGripperBase(GripperModel):
    """
    Two-Finger WSG Finray Gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/wsg50_finray_milibar_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.055, -0.055])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["left_finger_collision", "left_finger_pad_collision"],
            "right_finger": ["right_finger_collision", "right_finger_pad_collision"],
            "left_fingerpad": ["left_finger_pad_collision"],
            "right_fingerpad": ["right_finger_pad_collision"],
        }


class WSGGripper(WSGGripperBase):
    """
    Modifies WSGGripperBase to only take one action.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1
