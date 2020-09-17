# import math
import operator
from functools import reduce

import numpy as np
import gym
from gym import spaces
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
# from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
# from gym import error, spaces, utils
# from .minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX


class ViewWrapper(gym.core.ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env):
        super().__init__(env)
        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)
        # dtype=''uint8
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(imgSize,),
            dtype='float32'
        )
        env.unwrapped.spec.reward_threshold = 0.945

    def observation(self, obs):
        image = obs['image']
        obs = image.flatten()
        return obs


class GridWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact flattened grid encoding
    """

    def __init__(self, env):
        super().__init__(env)
        # self.shape = (self.env.width, self.env.height, 3)
        self.shape = (self.env.width*self.env.height*3,)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=self.shape,  # number of cells
            dtype='float32'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        full_grid = full_grid.reshape(-1)
        return full_grid
