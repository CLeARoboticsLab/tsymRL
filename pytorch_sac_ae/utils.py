import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
from typing import Tuple, Union

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

class TsymReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, percent_tsym, percent_sampling, phase_percent):
        self.capacity = capacity
        self.tsym_capacity = round(capacity * percent_tsym/100)
        self.batch_size = batch_size
        self.device = device
        self.percent_sampling = percent_sampling
        self.phase_percent = phase_percent

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.tsym_obses = np.empty((self.tsym_capacity, *obs_shape), dtype=obs_dtype)
        self.tsym_next_obses = np.empty((self.tsym_capacity, *obs_shape), dtype=obs_dtype)
        self.tsym_actions = np.empty((self.tsym_capacity, *action_shape), dtype=np.float32)
        self.tsym_rewards = np.empty((self.tsym_capacity, 1), dtype=np.float32)
        self.tsym_not_dones = np.empty((self.tsym_capacity, 1), dtype=np.float32)

        self.idx = 0
        self.tsym_idx = 0

        self.last_save = 0
        self.full = False
        self.tsym_full = False


    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_tsym(self, obs, action, reward, next_obs, done):

        if self.percent_sampling != "phase_in": # add to buffer normally

            np.copyto(self.tsym_obses[self.tsym_idx], obs)
            np.copyto(self.tsym_actions[self.tsym_idx], action)
            np.copyto(self.tsym_rewards[self.tsym_idx], reward)
            np.copyto(self.tsym_next_obses[self.tsym_idx], next_obs)
            np.copyto(self.tsym_not_dones[self.tsym_idx], not done)

            self.tsym_idx = (self.tsym_idx + 1) % self.tsym_capacity
            self.tsym_full = self.tsym_full or self.tsym_idx == 0

        else:
            if self.idx >= int(self.phase_percent/100 * self.capacity): # phase in after some number of steps relative to capacity of buffer
                np.copyto(self.tsym_obses[self.tsym_idx], obs)
                np.copyto(self.tsym_actions[self.tsym_idx], action)
                np.copyto(self.tsym_rewards[self.tsym_idx], reward)
                np.copyto(self.tsym_next_obses[self.tsym_idx], next_obs)
                np.copyto(self.tsym_not_dones[self.tsym_idx], not done)

                self.tsym_idx = (self.tsym_idx + 1) % self.tsym_capacity
                self.tsym_full = self.tsym_full or self.tsym_idx == 0

    def sample(self):
        # idxs = np.random.randint(
        #     0, self.capacity + self.tsym_capacity if self.full and self.tsym_full else self.idx + self.tsym_idx, size=self.batch_size
        # )


        if self.percent_sampling == "even":
            # Sample randomly in equal proportion across each individual replay buffer 
            idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size=int(self.batch_size/2)
            )

            tsym_idxs = np.random.randint(
                0, self.tsym_capacity if self.tsym_full else self.tsym_idx, size=int(self.batch_size/2)
            )
        # else: # natural sampling
            # Sample randomly across both replay buffers
            # BROKEN!!! see tsym_idx in else statement reseting on each full hit. Need to rethink
            # full_idxs = np.random.randint(
            #     0, self.capacity + self.tsym_capacity if self.full and self.tsym_full else self.idx + self.tsym_idx, size=self.batch_size
            # )
            # idxs = full_idxs[full_idxs < self.capacity]
            # tsym_idxs = full_idxs[full_idxs >= self.capacity] - self.capacity #new way
            # tsym_idxs = full_idxs[full_idxs >= self.capacity] #old way

        elif self.percent_sampling == "natural": # natural sampling
            max_base_sample = self.capacity if self.full else self.idx
            max_tsym_sample = self.tsym_capacity if self.tsym_full else self.tsym_idx

            full_idxs = np.random.randint(0, max_base_sample + max_tsym_sample, size=self.batch_size)
            idxs = full_idxs[full_idxs < max_base_sample]
            tsym_idxs = full_idxs[full_idxs >= max_base_sample] - max_base_sample #new way


        elif self.percent_sampling == "phase_in": # phase in after step sampling

            if self.idx > self.phase_percent/100 * self.capacity: # phase in after some number of steps relative to capacity of buffer
                max_base_sample = self.capacity if self.full else self.idx
                max_tsym_sample = self.tsym_capacity if self.tsym_full else self.tsym_idx

                full_idxs = np.random.randint(0, max_base_sample + max_tsym_sample, size=self.batch_size)
                idxs = full_idxs[full_idxs < max_base_sample]
                tsym_idxs = full_idxs[full_idxs >= max_base_sample] - max_base_sample #new way
            else:
                # draw from non time symmetric buffer
                idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size)
                tsym_idxs = np.empty(0, dtype=np.int8)

        else: # phase out with step

            if self.idx < 200000:
                max_base_sample = self.capacity if self.full else self.idx
                max_tsym_sample = self.tsym_capacity if self.tsym_full else self.tsym_idx

                full_idxs = np.random.randint(0, max_base_sample + max_tsym_sample, size=self.batch_size)
                idxs = full_idxs[full_idxs < max_base_sample]
                tsym_idxs = full_idxs[full_idxs >= max_base_sample] - max_base_sample #new way
            else:
                # Suddenly phase out time symmetric data after set number of steps
                idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size)
                tsym_idxs = np.empty(0, dtype=np.int8)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        tsym_obses = torch.as_tensor(self.tsym_obses[tsym_idxs], device=self.device).float()
        tsym_actions = torch.as_tensor(self.tsym_actions[tsym_idxs], device=self.device)
        tsym_rewards = torch.as_tensor(self.tsym_rewards[tsym_idxs], device=self.device)
        tsym_next_obses = torch.as_tensor(
            self.tsym_next_obses[tsym_idxs], device=self.device
        ).float()
        tsym_not_dones = torch.as_tensor(self.tsym_not_dones[tsym_idxs], device=self.device)

        return torch.cat((obses, tsym_obses)), torch.cat((actions, tsym_actions)), torch.cat((rewards, tsym_rewards)), torch.cat((next_obses, tsym_next_obses)), torch.cat((not_dones, tsym_not_dones))

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

class DualReplayBuffer(object):
    """Buffer to store environment transitions in both image and proprioceptive state."""
    def __init__(self, pix_obs_shape, prop_obs_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        # obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.pix_obses = np.empty((capacity, *pix_obs_shape), dtype=np.uint8)
        self.prop_obses = np.empty((capacity, *prop_obs_shape), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, pix_obs, prop_obs):
        np.copyto(self.pix_obses[self.idx], pix_obs)
        np.copyto(self.prop_obses[self.idx], prop_obs)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        pix_obses = torch.as_tensor(self.pix_obses[idxs], device=self.device).float()
        prop_obses = torch.as_tensor(self.prop_obses[idxs], device=self.device).float()

        return pix_obses, prop_obses

    def save(self, save_dir):

        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.pix_obses[self.last_save:self.idx],
            self.prop_obses[self.last_save:self.idx],
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.pix_obses[start:end] = payload[0]
            self.prop_obses[start:end] = payload[1]
            self.idx = end
        print("Loaded ", end, " datapoints")



class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


import numpy as np

# import gymnasium as gym
# from gymnasium.error import DependencyNotInstalled
# from gymnasium.spaces import Box


# class ResizeObservation(gym.ObservationWrapper):
#     """Resize the image observation.

#     This wrapper works on environments with image observations. More generally,
#     the input can either be two-dimensional (AxB, e.g. grayscale images) or
#     three-dimensional (AxBxC, e.g. color images). This resizes the observation
#     to the shape given by the 2-tuple :attr:`shape`.
#     The argument :attr:`shape` may also be an integer, in which case, the
#     observation is scaled to a square of side-length :attr:`shape`.

#     Example:
#         >>> import gymnasium as gym
#         >>> from gymnasium.wrappers import ResizeObservation
#         >>> env = gym.make("CarRacing-v2")
#         >>> env.observation_space.shape
#         (96, 96, 3)
#         >>> env = ResizeObservation(env, 64)
#         >>> env.observation_space.shape
#         (64, 64, 3)
#     """

#     # def __init__(self, env: gym.Env, shape: tuple[int, int] | int) -> None:
    
#     def __init__(self, env: gym.Env, shape: Union[Tuple[int, int], int]) -> None:
#         """Resizes image observations to shape given by :attr:`shape`.

#         Args:
#             env: The environment to apply the wrapper
#             shape: The shape of the resized observations
#         """
#         super().__init__(env)
#         if isinstance(shape, int):
#             shape = (shape, shape)
#         assert len(shape) == 2 and all(
#             x > 0 for x in shape
#         ), f"Expected shape to be a 2-tuple of positive integers, got: {shape}"

#         self.shape = tuple(shape)

#         assert isinstance(
#             env.observation_space, Box
#         ), f"Expected the observation space to be Box, actual type: {type(env.observation_space)}"
#         dims = len(env.observation_space.shape)
#         assert (
#             dims == 2 or dims == 3
#         ), f"Expected the observation space to have 2 or 3 dimensions, got: {dims}"


#         # Have the rgb dim first
#         obs_shape = env.observation_space.shape[2:] + self.shape
#         self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

#     def observation(self, observation):
#         """Updates the observations by resizing the observation to shape given by :attr:`shape`.

#         Args:
#             observation: The observation to reshape

#         Returns:
#             The reshaped observations

#         Raises:
#             DependencyNotInstalled: opencv-python is not installed
#         """
#         try:
#             import cv2
#         except ImportError as e:
#             raise DependencyNotInstalled(
#                 "opencv (cv2) is not installed, run `pip install gymnasium[other]`"
#             ) from e

#         observation = cv2.resize(
#             observation.transpose(1, 2, 0), self.shape[::-1], interpolation=cv2.INTER_AREA
#         )
#         return observation.reshape(self.observation_space.shape)



# class GymnasiumFrameStack(gym.Wrapper):
#     def __init__(self, env, k):
#         gym.Wrapper.__init__(self, env)
#         self._k = k
#         self._frames = deque([], maxlen=k)
#         shp = env.observation_space.shape
#         self.observation_space = gym.spaces.Box(
#             low=0,
#             high=1,
#             shape=((shp[0] * k,) + shp[1:]),
#             dtype=env.observation_space.dtype
#         )
#         # self._max_episode_steps = env._max_episode_steps

#     def reset(self):
#         obs, _ = self.env.reset()
#         for _ in range(self._k):
#             self._frames.append(obs)
#         return self._get_obs(), _

#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         self._frames.append(obs)
#         return self._get_obs(), reward, terminated, truncated, info

#     def _get_obs(self):
#         assert len(self._frames) == self._k
#         return np.concatenate(list(self._frames), axis=0)


from segment_tree import SumSegmentTree, MinSegmentTree

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
    
# Open AI implementations of replay buffer and prioritized experience replay from here: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
class OpenAIReplayBuffer(object):
    def __init__(self, capacity, batch_size, device):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = capacity
        self._next_idx = 0
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, not done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)

        # Convert to tensors and return
        return torch.as_tensor(np.array(obses_t), device=self.device).float(), torch.as_tensor(np.array(actions), device=self.device).float(), torch.as_tensor(np.array(rewards), device=self.device).unsqueeze(1).float(), torch.as_tensor(np.array(obses_tp1), device=self.device).float(), torch.as_tensor(np.array(dones), device=self.device).unsqueeze(1).float()
    
    def sample(self):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(self.batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(OpenAIReplayBuffer):
    def __init__(self, capacity, batch_size, device, alpha = 0.6):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(capacity, batch_size, device)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / self.batch_size
        for i in range(self.batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional()

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)