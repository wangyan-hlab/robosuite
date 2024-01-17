from .base import REGISTERED_ENVS, MujocoEnv
from robosuite.environments.manipulation.twist_lock import TwistLock

REGISTERED_ENVS[TwistLock.__name__] = TwistLock
ALL_ENVIRONMENTS = REGISTERED_ENVS.keys()
