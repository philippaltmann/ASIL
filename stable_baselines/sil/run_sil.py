import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import SIL

env = gym.make('CartPole-v1')

model = SIL(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("sil_cartpole")
