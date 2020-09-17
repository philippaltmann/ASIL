import os
import sys
import numpy as np
import gym_minigrid
from stable_baselines import ASIL
from stable_baselines.common import make_vec_env
from stable_baselines.common.grid_wrapper import ViewWrapper
from stable_baselines.common.evaluation import evaluate_policy

assert len(sys.argv) == 3, 'Please specify the environment and model path'
env_name = sys.argv[1]
model_dir = sys.argv[2]

env = make_vec_env(env_name, n_envs=1, wrapper_class=ViewWrapper)
obs = env.reset()

model = ASIL.load(os.path.join(model_dir, 'post'))

# Evaluate the agent
overall = []
r = []
l = []
fails = []
dies = []
obs = env.reset()
episodes = 1000
episode = 0
while episode < episodes:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # env.render()
    if len(info[0]) > 0:
        i = info[0]['episode']
        episode += 1
        if i['r'] == 0:
            if i['l'] == 252:
                fails.append(i['l'])
            else:
                dies.append(i['l'])
            print(f"Failed after {i['l']} steps")
        else:
            r.append(i['r'])
            l.append(i['l'])
            print(f"Succeeded with {i['r']} | {i['l']}")
        overall.append(i['r'])


print(np.mean(r))
print(np.mean(l))
print(f"Succeeded {len(r)} times with mean reward {np.mean(r)} after {np.mean(l)} steps" )
print(f"Failed {len(fails)} times, \nDied {len(dies)} times afer {np.mean(dies)} steps")
print(f"Overall Mean Reward Mean: {np.mean(overall)}")
