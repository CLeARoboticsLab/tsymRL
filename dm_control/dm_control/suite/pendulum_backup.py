# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================



import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np


_DEFAULT_TIME_LIMIT = 20
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(_ANGLE_BOUND))
SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('pendulum.xml'), common.ASSETS

@SUITE.add('benchmarking')
def swingup(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns pendulum swingup task ."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = SwingUp(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Pendulum domain."""

  def pole_vertical(self):
    """Returns vertical (z) component of pole frame."""
    return self.named.data.xmat['pole_1', 'zz']

  def angular_velocity(self):
    """Returns the angular velocity of the pole."""
    return self.named.data.qvel['hinge_1'].copy()

  def pole_orientation(self):
    """Returns both horizontal and vertical components of pole frame."""
    return self.named.data.xmat['pole_1', ['zz', 'xz']]


class SwingUp(base.Task):
  """A Pendulum `Task` to swing up and balance the pole."""

  def __init__(self, random=None):
    """Initialize an instance of `Pendulum`.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Pole is set to a random angle between [-pi, pi).

    Args:
      physics: An instance of `Physics`.

    """
    physics.named.data.qpos['hinge'] = self.random.uniform(-np.pi, np.pi)
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation.

    Observations are states concatenating pole orientation and angular velocity
    and pixels from fixed camera.

    Args:
      physics: An instance of `physics`, Pendulum physics.

    Returns:
      A `dict` of observation.
    """
    obs = collections.OrderedDict()
    obs['orientation'] = physics.pole_orientation()
    obs['velocity'] = physics.angular_velocity()
    return obs

  def get_reward(self, physics):

      upright = (physics.pole_vertical() + 1) / 2
      small_control = rewards.tolerance(physics.control(), margin=1,
                                        value_at_margin=0,
                                        sigmoid='quadratic').min()
      small_control = (4 + small_control) / 5
      small_velocity = rewards.tolerance(physics.angular_velocity(), margin=5).min()
      small_velocity = (1 + small_velocity) / 2
      return upright.mean() * small_control * small_velocity
  
  def get_rev_reward(self, physics, action):
    """Returns a sparse or a smooth reversed time reward, as specified in the constructor."""
    upright = (physics.pole_vertical() + 1) / 2
    small_control = rewards.tolerance(action, margin=1,
                                      value_at_margin=0,
                                      sigmoid='quadratic').min()
    small_control = (4 + small_control) / 5
    small_velocity = rewards.tolerance(physics.angular_velocity(), margin=5).min()
    small_velocity = (1 + small_velocity) / 2
    return upright.mean() * small_control * small_velocity