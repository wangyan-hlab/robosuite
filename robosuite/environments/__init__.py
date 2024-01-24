from .base import REGISTERED_ENVS, MujocoEnv
from robosuite.environments.manipulation.twist_lock import TwistLock
from robosuite.environments.manipulation.twist_lock_12 import TwistLock12

REGISTERED_ENVS[TwistLock.__name__] = TwistLock
REGISTERED_ENVS[TwistLock12.__name__] = TwistLock12
ALL_ENVIRONMENTS = REGISTERED_ENVS.keys()
