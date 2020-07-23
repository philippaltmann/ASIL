import os
import gym
from stable_baselines import ASIL
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecVideoRecorder, DummyVecEnv

# multiprocess environment
env_name = 'CartPole-v1'
# env_name = 'BipedalWalker-v3'
# env_name = 'Swimmer-v2'

use_gasil = True  # TODO read from kwargs in main
algorithm = "ASIL" if use_gasil else "PPO2"

# Algorithm Params
total_timesteps = 1000000  # CartPole
# total_timesteps = 2000000  # BipedalWalker
# total_timesteps = 5000000  # Swimmer

sil_samples = 512
sil_alpha = 1.0

run_index = 1
base_path = "test/{}/{}/{}/".format(env_name, algorithm, run_index)
# base_path = "new_logs/{}/{}/{}/".format(env_name, algorithm, run_index)
while os.path.exists(base_path):
    run_index += 1
    base_path = "test/{}/{}/{}/".format(env_name,  algorithm, run_index)
    # base_path = "new_logs/{}/{}/{}/".format(env_name,  algorithm, run_index)

print("Logging to {}".format(base_path))
tb_folder = '{}tb/'.format(base_path)
model_dir = '{}model/'.format(base_path)
os.makedirs(model_dir)

# Video Settings
video_length = 400  # 200   # The length of saved videos
video_factor = 50000  # 10000  # 500000  # The Frequency of saving videos
video_folder = '{}videos/'.format(base_path)

env = make_vec_env(env_name, n_envs=1)  # n_envs=4

# Record the video starting at the first step
# env = VecVideoRecorder(
#     env, video_folder, video_length=video_length, name_prefix="agent",
#     record_video_trigger=lambda x: (x + video_length) % video_factor == 0
# )

# TODO flag for using extension or not (also causing naming to be PPO2)
model = ASIL(MlpPolicy, env, verbose=1, tensorboard_log=tb_folder,
             env_name=env_name, use_gasil=use_gasil, terminate_on_solve=True,
             sil_samples=sil_samples, sil_alpha=sil_alpha)

model.learn(total_timesteps=total_timesteps)

# TODO: Fix saving % loading
print("Done Training. Saving model...")
model.save(os.path.join(model_dir, 'post'))

print("Rendering Final Video...")

for step in range(2):
    # Record the video starting at the first step
    env = make_vec_env(env_name, n_envs=1)
    env = VecVideoRecorder(env, video_folder, video_length=510,
                           record_video_trigger=lambda x: x == 0,
                           name_prefix="final_{}".format(step))
    obs = env.reset()
    for _ in range(510):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
    env.close()

del model  # delete trained model to demonstrate loading

# Load the trained agent
model = ASIL.load(os.path.join(model_dir, 'post'))

for step in range(2):
    # Record the video starting at the first step
    env = make_vec_env(env_name, n_envs=1)
    env = VecVideoRecorder(env, video_folder, video_length=510,
                           record_video_trigger=lambda x: x == 0,
                           name_prefix="reload_{}".format(step))
    obs = env.reset()
    for _ in range(510):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
    env.close()






# model.save("ppo2_cartpole")
#
# del model # remove to demonstrate saving and loading
#
# model = PPO2.load("ppo2_cartpole")
#
# # Enjoy trained agent
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

#https://github.com/tensorflow/tensorboard/pull/2126/files

# xvfb-run -s "-screen 0 1400x900x24" python -m baselines.run --alg=ppo2 --env=CartPole-v1 --network=mlp --num_timesteps=1e6 --save_video_interval=100000
# xvfb-run -s "-screen 0 1400x900x24" python run_gasil.py
# pip install -U tensorflow==1.14.0
