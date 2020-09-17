#!/usr/bin/env python3

import time
import math
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window


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

from gym.envs.registration import register

register(
    id='MiniGrid-DistShift1-v1',
    entry_point='gym_minigrid.envs:DistShift1',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 100},
    reward_threshold=0.9,
)
class MyFlatObsWrapper(gym.core.ObservationWrapper):
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
        print("Testing Wrapped Env")
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        # print(full_grid)
        # for x in full_grid:
        #     for y in x:
        #         print(y[0], end="")
        #     print("\n")

    def observation(self, obs):
        image = obs['image']
        obs = image.flatten()
        # env = self.unwrapped
        # full_grid = env.grid.encode()
        # print(full_grid)
        return obs


def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    # print('step=%s, reward=%.2f' % (env.step_count, reward))
    # print(obs)
    # if len(info) > 0:
    if done:
        print(env.__dict__)
        print(env.spec.__dict__)
        print(env.actions.__dict__)
        print(env.unwrapped)

        print(reward)
    print(1-0.9*(env.step_count/256)) # 252
    print(env.step_count)

    # print(len(obs))
    # for f in obs:
    #     print(int(f), end=" ")
    # for x in range(7):
    #     for y in range(7):
    #         print(int(obs[x*7+y]), end="")
    #         print(int(obs[x*7+y+1]), end="")
    #         print(int(obs[x*7+y+2]), end=" ")
    #     print("")
    # env = env.unwrapped
    # full_grid = env.grid.encode()
    # full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
    #     OBJECT_TO_IDX['agent'],
    #     COLOR_TO_IDX['red'],
    #     env.agent_dir
    # ])
    #
    # print(env.agent_dir)
    #
    # # print(full_grid)
    # for x in full_grid:
    #     for y in x:
    #         print(math.floor(y[0]/2), end="")
    #     print("")
    #
    # image = obs['image']
    # for x in image:
    #     for y in x:
    #         print(y, end="")
    #     print("")

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--env",
#     help="gym environment to load",
#     default='MiniGrid-MultiRoom-N6-v0'
# )
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-DistShift1-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

args = parser.parse_args()

env = gym.make(args.env)
# env = MyFlatObsWrapper(env)

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
