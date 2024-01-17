from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import TwistLockObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler


class TwistLock(SingleArmEnv):
    """
    This class corresponds to the twist lock unlocking task for a single robot arm.

    Args:

    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        mount_types="default",
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.x = np.random.uniform(-0.01, 0.01)
        self.y = np.random.uniform(-0.2, 0.2)
        self.rotation = np.random.uniform(-np.pi / 2.0 - 0.25, -np.pi / 2.0 + 0.25)

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the lock is opened

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.25], proportional to the distance between lock handle and robot arm
            - Rotating: in [0, 0.25], proportional to angle rotated by lock handled

        Note that a successfully completed task (lock opened) will return 1.0 irregardless of whether the environment
        is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # else, we consider only the case if we're using shaped rewards
        elif self.reward_shaping:
            # Add reaching component
            dist = np.linalg.norm(self._gripper_to_handle)
            reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
            reward += reaching_reward

            handle_qpos = self.sim.data.qpos[self.handle_qpos_addr]
            reward += np.clip(0.25 * np.abs(handle_qpos / (0.5 * np.pi)), -0.25, 0.25)

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )

        self.twistlocks = []

        # initialize objects of interest
        self.twistlock = TwistLockObject(
            name="TwistLock",
            friction=1.0,
            damping=0.1,
            container=True
        )

        # Create placement initializer
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="TwistLockSampler",
                    mujoco_objects=self.twistlock,
                    x_range=[-0.01, 0.01],
                    y_range=[-0.2, 0.2],
                    rotation=(-np.pi / 2.0 - 0.25, -np.pi / 2.0 + 0.25),
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=0.075,
                )
            )
        # Reset sampler before adding any new samplers / objects
        self.placement_initializer.reset()

        self.twistlocks.append(self.twistlock)
        # Add the twistlock to the placement initializer
        if isinstance(self.placement_initializer, SequentialCompositeSampler):
            # assumes we have two objects so we add them to the two samplers
            self.placement_initializer.add_objects_to_sampler(
                sampler_name="TwistLockSampler",
                mujoco_objects=self.twistlock
            )
        else:
            # This is assumed to be a flat sampler, so we just add all objects to this sampler
            self.placement_initializer.add_objects(self.twistlock)

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.twistlocks,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.object_body_ids = dict()
        self.object_body_ids["twistlock_casing"] = self.sim.model.body_name2id(self.twistlock.casing_body)
        self.object_body_ids["twistlock_lock"] = self.sim.model.body_name2id(self.twistlock.lock_body)
        self.twistlock_handle_site_id = self.sim.model.site_name2id(self.twistlock.important_sites["handle"])
        self.lockjoint_qpos_addr = self.sim.model.get_joint_qpos_addr(self.twistlock.joints[-1])
        self.handle_qpos_addr = self.sim.model.get_joint_qpos_addr(self.twistlock.joints[-1])

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # Define sensor callbacks
            @sensor(modality=modality)
            def twistlock_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.object_body_ids["twistlock_lock"]])

            @sensor(modality=modality)
            def handle_pos(obs_cache):
                return self._handle_xpos

            @sensor(modality=modality)
            def twistlock_to_eef_pos(obs_cache):
                return (
                    obs_cache["twistlock_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "twistlock_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def handle_to_eef_pos(obs_cache):
                return (
                    obs_cache["handle_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "handle_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def lockjoint_qpos(obs_cache):
                return np.array([self.sim.data.qpos[self.lockjoint_qpos_addr]])

            sensors = [twistlock_pos, handle_pos, twistlock_to_eef_pos, handle_to_eef_pos, lockjoint_qpos]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            twistlock_pos, twistlock_quat, _ = object_placements[self.twistlock.name]
            twistlock_body_id = self.sim.model.body_name2id(self.twistlock.root_body)
            self.sim.model.body_pos[twistlock_body_id] = twistlock_pos
            self.sim.model.body_quat[twistlock_body_id] = twistlock_quat

    def _check_success(self):
        """
        Check if lock has been opened and lifted.

        Returns:
            bool: True if lock has been opened and lifted
        """
        lockjoint_qpos = self.sim.data.qpos[self.lockjoint_qpos_addr]
        lock_height = self.sim.data.body_xpos[self.object_body_ids["twistlock_lock"]][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        # lock is opened and higher than the table top above a margin
        return lockjoint_qpos > 1.48 and lock_height > table_height + 0.2

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the lock handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the door handle
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.twistlock.important_sites["handle"], target_type="site"
            )

    @property
    def _handle_xpos(self):
        """
        Grabs the position of the lock handle.

        Returns:
            np.array: Lock handle (x,y,z)
        """
        return self.sim.data.site_xpos[self.twistlock_handle_site_id]

    @property
    def _gripper_to_handle(self):
        """
        Calculates distance from the gripper to the lock handle.

        Returns:
            np.array: (x,y,z) distance between handle and eef
        """
        return self._handle_xpos - self._eef_xpos
