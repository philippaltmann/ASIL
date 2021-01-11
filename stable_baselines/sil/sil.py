import gym
import tensorflow as tf
import numpy as np
from collections import deque

from stable_baselines import logger
from stable_baselines.ppo2 import PPO2
from stable_baselines.ppo2.ppo2 import swap_and_flatten
from stable_baselines.sil.self_imitation import SelfImitation

from stable_baselines.common import tf_util
from stable_baselines.common.callbacks import BaseCallback, CallbackList


class SIL(PPO2):
    """
    Self-Imitation Learning (SIL)

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param noptepochs: (int) Number of epoch when optimizing the surrogate
    :param cliprange: (float or callable) Clipping parameter, it can be a function
    :param cliprange_vf: (float or callable) Clipping parameter for the value function, it can be a function.
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling.
        To deactivate value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.

    :param sil_update: (int) The frequency of learning the Discriminator
    :param sil_value: Value for SIL.
    :param sil_alpha: (float) the weight of the Self-Imitation Reward.
    :param sil_beta: (float) the weight of the Self-Imitation Reward.
    """

    def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, threshold=None,
                 terminate_on_solve=False, max_episodes=1000, sil_update=1,
                 sil_value=0.01, sil_alpha=0.6, sil_beta=0.1, **kwargs):
        super().__init__(policy, env, gamma=gamma, n_steps=n_steps,
                         ent_coef=ent_coef, learning_rate=learning_rate,
                         vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                         lam=lam, nminibatches=nminibatches,
                         noptepochs=noptepochs, cliprange=cliprange,
                         cliprange_vf=cliprange_vf, verbose=verbose,
                         tensorboard_log=tensorboard_log,
                         _init_setup_model=False, policy_kwargs=policy_kwargs,
                         full_tensorboard_log=full_tensorboard_log,
                         seed=seed, n_cpu_tf_sess=n_cpu_tf_sess,  **kwargs)

        # SIL Params
        self.sil = None
        self.sil_model = None
        self.sil_update = sil_update
        print(dir(self.env.envs[0]))
        # Using Identity function for processing SIL rewards & observations

        self.sil_value = sil_value
        self.sil_alpha = sil_alpha
        self.sil_beta = sil_beta

        # Keep Track of Episode Reward (set, update, log from Callback)
        self.real_reward = []  # Env Reward Buffer for Adapted Rollouts
        self.d_reward = []     # Discriminator Reward Buffer for Adapted Rollouts
        self.action_probs = []
        self.total_episodes = 0
        self.current_episode = 0
        self.moving_reward = deque([], maxlen=100)
        self.terminate_on_solve = terminate_on_solve
        self.max_episodes = max_episodes
        self.stop = False
        try:
            self.threshold = float(threshold)
        except Exception:
            self.threshold = None
        if terminate_on_solve:
            logger.log("Stopping at %s" % self.threshold)

        if _init_setup_model:
            self.setup_model()
        print("Done With setup")

    def setup_model(self):
        logger.log("Setting Up PPO Model..")
        super().setup_model()
        logger.log("Adding SIL Dependencies..")
        with self.graph.as_default():
            with tf.compat.v1.variable_scope("sil_model", reuse=True, custom_getter=tf_util.outer_scope_getter("sil_model")):
                sil_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, None,
                            None, reuse=False, **self.policy_kwargs)

            self.params = tf.compat.v1.trainable_variables()

            # Self-Imitation learning
            self.sil = SelfImitation(sil_model.obs_ph, sil_model.value_fn,
                    sil_model.proba_distribution.entropy(), sil_model.value,
                    sil_model.proba_distribution.neglogp, self.action_space,
                    n_env=self.n_envs, n_update=self.sil_update,
                    w_value=self.sil_value, w_entropy=self.ent_coef,
                    gamma=self.gamma, max_steps=50000, max_nlogp=100,
                    alpha=self.sil_alpha, beta=self.sil_beta)

            self.sil.set_loss_weight(0.1)

            trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
            self.sil.build_train_op(self.params, trainer, self.learning_rate_ph)

            # FROM ASIL
            # self.summary = tf.compat.v1.summary.merge_all()  # TODO test if needed / usable
            # self.adversary.setup_trainer()
            tf.compat.v1.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101
            self.sil_model = sil_model

    def sil_train(self, cur_lr):
        return self.sil.train(self.sess, cur_lr)

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="SIL",
              reset_num_timesteps=True):
        print("Learn not defined!")
        env_name = self.env.envs[0].unwrapped.spec.id
        if self.threshold is None:
            self.threshold = self.env.envs[0].unwrapped.spec.reward_threshold
        callbacks = []
        if isinstance(callback, CallbackList):
            callbacks = callbacks + callback.callbacks
            print(callbacks)
        learning = SelfImitationCallback(verbose=self.verbose)
        callback = CallbackList(callbacks + [learning])
        return super().learn(
            total_timesteps, callback=callback, log_interval=log_interval,
            tb_log_name='', reset_num_timesteps=reset_num_timesteps)

    def get_hparams(self):
        return {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
            "cliprange_vf": self.cliprange_vf,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "sil_update": self.sil_update,
            "sil_value": self.sil_value,
            "sil_alpha": self.sil_alpha,
            "sil_beta": self.sil_beta
        }

    def save(self, save_path, cloudpickle=False):
        # verbose=0, tensorboard_log=None, _init_setup_model=True, full_tensorboard_log=False,
        data = self.get_hparams()
        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)


class SelfImitationCallback(BaseCallback):
    """
    A custom callback Updating the Self-Imitation Extension

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(SelfImitationCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        observations = self.locals['obs'].copy()
        actions = self.locals['actions'].copy()
        dones = self.locals['dones'].copy()
        rewards = self.locals['rewards'].copy()
        self.model.sil.step(observations, actions, rewards, dones)


    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        lrnow = self.locals['lr_now']
        sil_loss, sil_adv, sil_samples, sil_nlogp = self.model.sil_train(lrnow)

        # Add Stats to logger
        logger.logkv("100_mean_reward", np.mean(self.model.moving_reward))
        logger.logkv("sil_samples", sil_samples)

        # Get Variables
        ep_infos = self.locals['ep_infos']
        writer = self.locals['writer']

        # Write Summary if writer available
        if writer is None:
            return

        for info in ep_infos:
            self.model.total_episodes += 1
            self.model.current_episode += info['l']
            self.model.moving_reward.append(info['r'])
            writer.add_summary(tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(tag='rewards/episode_length',
                                 simple_value=info['l']),
                tf.compat.v1.Summary.Value(tag='rewards/environment_reward',
                                 simple_value=info['r']),
                tf.compat.v1.Summary.Value(tag='rewards/mean_reward',
                                 simple_value=np.mean(self.model.moving_reward)),
                tf.compat.v1.Summary.Value(tag='rewards/number_episodes',
                                 simple_value=self.model.total_episodes)
            ]), self.model.current_episode)

        # Write Detailed SIL Summary
        writer.add_summary(tf.compat.v1.Summary(value=[
            tf.compat.v1.Summary.Value(tag='sil/loss', simple_value=sil_loss),
            tf.compat.v1.Summary.Value(tag='sil/adv', simple_value=sil_adv),
            tf.compat.v1.Summary.Value(tag='sil/samples', simple_value=sil_samples),
            tf.compat.v1.Summary.Value(tag='sil/nlogp', simple_value=np.mean(sil_nlogp)),
        ]), self.num_timesteps)
