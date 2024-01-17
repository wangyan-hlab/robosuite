from robosuite.models.arenas import TableArena
from robosuite.utils.mjcf_utils import array_to_string, find_elements
import numpy as np
from robosuite.utils.transform_utils import quat_multiply

class ContainerArena(TableArena):
    """
    Workspace that contains a tabletop with a fixed container which can be used in Twist-lock locking task.

    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
    """

    def __init__(
        self,
        table_full_size=(0.45, 0.69, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0),
    ):
        super().__init__(
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_offset,
            xml="arenas/container_arena.xml",
        )

        # Get references to container body
        self.container_body = self.worldbody.find("./body[@name='container']")

    def set_container_pos(self, pos):
        """
        Set the container position given the pos value

        Args:
            pos(3-array):[x,y,z]
        """
        self.container_body.set("pos", array_to_string(pos))

    def set_container_quat(self, rot_angle, init_quat, rotation_axis="z"):
        """
        Set the container quaternion given the rotation angle,
        the initial quaternion, and the rotation axia

        Args:
            rot_angle: a degree in radian
            init_quat(4-array): [w,x,y,z]
            rotation_axis(str): "x" ,"y" or "z"
        """
        # Return angle based on axis requested
        if rotation_axis == "x":
            quat = np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
        elif rotation_axis == "y":
            quat = np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
        elif rotation_axis == "z":
            quat = np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(rotation_axis)
            )
        quat = quat_multiply(quat, init_quat)
        self.container_body.set("quat", array_to_string(quat))

    @property
    def container_pos(self):
        pos = self.container_body.get("pos")
        pos = np.array(list(map(float, pos.split())))
        return pos

    @property
    def container_quat(self):
        quat = self.container_body.get("quat")
        quat = np.array(list(map(float, quat.split())))
        return quat
