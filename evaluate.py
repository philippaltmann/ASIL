import os
import sys
from stable_baselines import ASIL
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecVideoRecorder

env_arg = sys.argv[1]  # CartPole / Walker
env_name = None
total_timesteps = None
max_episodes = None
sil_alpha = float(sys.argv[2])
sil_samples = int(sys.argv[3])
terminate_on_solve = len(sys.argv) == 5

# Usage:
# # xvfb-run -s "-screen 0 1400x900x24" python3 evaluate.py CartPole 1 512 T

if env_arg == 'CartPole':
    env_name = 'CartPole-v1'
    total_timesteps = 1000000
    max_episodes = 1250
elif env_arg == 'Walker':
    env_name = 'BipedalWalker-v3'
    total_timesteps = 2000000
    max_episodes = total_timesteps  # => not stopping if not solved
else:
    print("No Vaild Environment specified")
    sys.exit()

use_gasil = True  # TODO read from kwargs in main

# Video Settings
video_length = 400  # 200   # The length of saved videos
video_factor = 50000  # 10000  # 500000  # The Frequency of saving videos

run = 1
base_path = "eval/{}/{}_{}/{}/".format(env_arg, sil_alpha, sil_samples, run)
# base_path = "new_logs/{}/{}/{}/".format(env_name, algorithm, run_index)
while os.path.exists(base_path):
    run += 1
    base_path = "eval/{}/{}_{}/{}/".format(env_arg, sil_alpha, sil_samples, run)
    # base_path = "new_logs/{}/{}/{}/".format(env_name,  algorithm, run_index)

tb_folder = '{}tb/'.format(base_path)
model_dir = '{}model/'.format(base_path)
video_folder = '{}videos/'.format(base_path)
print("Logging to {}".format(base_path))
os.makedirs(model_dir)

env = make_vec_env(env_name, n_envs=1)  # n_envs=4

# TODO flag for using extension or not (also causing naming to be PPO2)
model = ASIL(MlpPolicy, env, verbose=1, tensorboard_log=tb_folder,
             use_gasil=use_gasil, sil_alpha=sil_alpha, sil_samples=sil_samples,
             terminate_on_solve=terminate_on_solve, max_episodes=max_episodes)

model.learn(total_timesteps=total_timesteps)

# TODO: Fix saving % loading
print("Done Training. Saving model...")
model.save(os.path.join(model_dir, 'post'))

print("Rendering Final Video...")

# Record the video starting at the first step
env = make_vec_env(env_name, n_envs=1)
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

env = make_vec_env(env_name, n_envs=1)
env = VecVideoRecorder(env, video_folder, video_length=510,
                       record_video_trigger=lambda x: x == 0,
                       name_prefix="reload")
obs = env.reset()
for _ in range(510):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
env.close()
