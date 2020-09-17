import os
import sys
from stable_baselines import ASIL
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecVideoRecorder
import gym_minigrid
from stable_baselines.common.grid_wrapper import GridWrapper, ViewWrapper


env_arg = sys.argv[1]  # CartPole / Walker
env_name = None
wrapper_class = None
total_timesteps = None
max_episodes = None
sil_alpha = float(sys.argv[2])
sil_update = int(sys.argv[3])
sil_samples = int(sys.argv[4])
run = int(sys.argv[5])
terminate_on_solve = len(sys.argv) == 7
threshold = sys.argv[6] if terminate_on_solve else None

adversary_step = 1
# Usage:
# xvfb-run -s "-screen 0 1400x900x24" python3 evaluate.py Grid 1.0 1 32 1 T

if env_arg == 'CartPole':
    env_name = 'CartPole-v1'
    total_timesteps = 1000000
    max_episodes = total_timesteps  # 1250
elif env_arg == 'Grid':
    # env_name = 'MiniGrid-Empty-5x5-v0'
    env_name = 'MiniGrid-Empty-8x8-v0'
    wrapper_class = GridWrapper
    total_timesteps = 500000
    max_episodes = total_timesteps  # => not stopping if not solved
elif env_arg == 'Grid2':
    # 'MiniGrid-FourRooms-v0' 'MiniGrid-LavaGapS5-v0' 'MiniGrid-SimpleCrossingS9N1-v0'
    env_name = 'MiniGrid-DistShift1-v0'
    wrapper_class = ViewWrapper
    total_timesteps = 500000
    max_episodes = total_timesteps  # => not stopping if not solved
elif env_arg == 'Grid3':
    # 'MiniGrid-FourRooms-v0' 'MiniGrid-LavaGapS5-v0'
    env_name = 'MiniGrid-SimpleCrossingS9N1-v0'
    wrapper_class = ViewWrapper
    total_timesteps = 2000000
    max_episodes = total_timesteps  # => not stopping if not solved
elif env_arg == 'Grid4':
    # 'MiniGrid-FourRooms-v0' 'MiniGrid-LavaGapS5-v0'
    env_name = 'MiniGrid-LavaCrossingS9N1-v0'
    wrapper_class = ViewWrapper
    total_timesteps = 1000000
    max_episodes = total_timesteps  # => not stopping if not solved
elif env_arg == 'Mountain':
    env_name = 'MountainCar-v0'  # 'MountainCarContinuous-v0'#
    total_timesteps = 2000000
    max_episodes = total_timesteps  # => not stopping if not solved
elif env_arg == 'Lander':
    env_name = 'LunarLander-v2'
    total_timesteps = 2000000
    max_episodes = total_timesteps  # => not stopping if not solved
elif env_arg == 'Walker':
    env_name = 'BipedalWalker-v3'
    total_timesteps = 5000000  # Early stopping at 300 mean reward
    max_episodes = total_timesteps  # => not stopping if not solved
else:
    print("No Vaild Environment specified")
    sys.exit()

use_gasil = True  # TODO read from kwargs in main

# run = 1
base_path = "eval/{}/{}_{}_{}/{}/".format(env_arg, sil_alpha, sil_update, sil_samples, run)
# base_path = "new_logs/{}/{}/{}/".format(env_name, algorithm, run_index)
# while os.path.exists(base_path):
#     run += 1
#     base_path = "eval/{}/{}_{}_{}/{}/".format(env_arg, sil_alpha, sil_update, sil_samples, run)
    # base_path = "new_logs/{}/{}/{}/".format(env_name,  algorithm, run_index)

model_dir = '{}model/'.format(base_path)
video_folder = '{}videos/'.format(base_path)
print("Logging to {}".format(base_path))
os.makedirs(model_dir)

# Adapt to Update Rate 1/2 => Update A twice for every rollout
if sil_update == 0:
    sil_update = 1
    adversary_step = 2

env = make_vec_env(env_name, n_envs=1, wrapper_class=wrapper_class)

# TODO flag for using extension or not (also causing naming to be PPO2)
# TODO remove terminate_on_solve flag, when using threeshold for ealy stopping
model = ASIL(MlpPolicy, env, verbose=1, tensorboard_log=base_path,
             sil_alpha=sil_alpha, adversary_step=adversary_step,
             sil_update=sil_update, sil_samples=sil_samples,
             terminate_on_solve=terminate_on_solve, threshold=threshold,
             max_episodes=max_episodes)

model.learn(total_timesteps=total_timesteps)

# TODO: Fix saving % loading
print("Done Training. Saving model...")
model.save(os.path.join(model_dir, 'post'))

print("Rendering Final Video...")

# Video Settings
video_length = 400  # 200   # The length of saved videos
video_factor = 50000  # 10000  # 500000  # The Frequency of saving videos

# Record the video starting at the first step
env = VecVideoRecorder(env, video_folder, video_length=510,
                       record_video_trigger=lambda x: x == 0,
                       name_prefix="final")
obs = env.reset()
for _ in range(510):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
env.close()
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = ASIL.load(os.path.join(model_dir, 'post'))

env = make_vec_env(env_name, n_envs=1, wrapper_class=wrapper_class)
env = VecVideoRecorder(env, video_folder, video_length=510,
                       record_video_trigger=lambda x: x == 0,
                       name_prefix="reload")
obs = env.reset()
for _ in range(510):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
env.close()
