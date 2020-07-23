"""
From /common/buffers, adapted Replay Buffer to store State-Action-Rewards only
Sorting items in storage by reward, removing wost on append, when buffer full
"""
import random
from typing import Optional, List, Union

import numpy as np

from stable_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from stable_baselines.common.vec_env import VecNormalize


class RewardBuffer(object):
    def __init__(self, size: int):
        """
        Closely related to stable_baselines' Replay Buffer
        Sorted by Reward instead of FIFO, not storing done or next_obs
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def storage(self):
        """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self) -> int:
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples: int) -> bool:
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self) -> int:
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs, action, reward, retrun):
        """
        add a new transition to the buffer if reward is k-highest

        :param obs: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the true reward of the transition
        :param reward: (float) the return used for sorting
        """
        data = (obs, action, reward, retrun)
        insert_index = len(self)
        # Insertion Sort
        # Move Insert Index forward while added reward is higher
        while insert_index > 0 and self._storage[insert_index - 1][3] <= retrun:
            insert_index -= 1
        self._storage.insert(insert_index, data)
        if len(self) > self.buffer_size:
            self._storage = self._storage[:self.buffer_size]

    def extend(self, obs, action, reward, retrun):
        """
        add a new batch of transitions to the buffer

        :param obs: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition

        Note: uses the same names as .add to keep compatibility with named argument passing
                but expects iterables and arrays with more than 1 dimensions
        """
        for o, a, r, s in zip(obs, action, reward, retrun):
            self.add(o, a, r, s)

    @staticmethod
    def _normalize_obs(obs: np.ndarray,
                       env: Optional[VecNormalize] = None) -> np.ndarray:
        """
        Helper for normalizing the observation.
        """
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray,
                          env: Optional[VecNormalize] = None) -> np.ndarray:
        """
        Helper for normalizing the reward.
        """
        if env is not None:
            return env.normalize_reward(reward)
        return reward

    def _encode_sample(self, idxes: Union[List[int], np.ndarray], env: Optional[VecNormalize] = None):
        obses, actions, rewards, returns = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs, action, reward, retrun = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            returns.append(retrun)
        return (self._normalize_obs(np.array(obses), env),
                np.array(actions),
                self._normalize_reward(np.array(rewards), env),
                self._normalize_reward(np.array(returns), env))

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes, env=env)
