import gym
import tensorflow as tf
import numpy as np

from stable_baselines import logger
from stable_baselines.ppo2 import PPO2
from stable_baselines.ppo2.ppo2 import swap_and_flatten

from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.gasil.adversary import TransitionClassifier
# from stable_baselines.common.mpi_adam import MpiAdam

from stable_baselines.gasil.buffer import RewardBuffer
# from stable_baselines.common import dataset
from stable_baselines.common.tf_util import total_episode_reward_logger
from tensorboard.plugins.hparams import api as hp



# from stable_baselines.common import explained_variance, ActorCriticRLModel, SetVerbosity, TensorboardWriter, dataset


class GASIL(PPO2):
    """
    Generative Adversarial Self-Imitation Learning (GASIL)

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

    :param use_gasil: (bool) Whether or not to use GASIL Reward for learning
    :param sil_samples: (int) Max Number of trajectories stored in SIL Buffer
    :param g_step: (int) number of steps to train policy in each epoch
        (Train Discriminator every after n policy updates)
    :param adversary_step: (int) number of steps to train discriminator in each epoch
    :param adversary_entcoeff: (float) the adversary entropy coefficient (1e-3)
    :param sil_alpha: (float) the weight of the Discriminator Reward.
        1 => just D (pure GASIL), 0 => just real reward (pure PPO)
    :param adversary_stepsize: (float) the Adversarys stepsize on update
    :param adversary_hidden_size: (int) the hidden dimension for the Discriminator Network (100)
    """
    # sil_update=1, sil_value=0.01, sil_alpha=0.6, sil_beta=0.1,

    def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None,
                 adversary_hidden_size=100, adversary_entcoeff=1e-3, sil_alpha=1.0,
                 g_step=3, adversary_step=1, adversary_stepsize=3e-4, sil_samples=512,
                 use_gasil=True, env_name="", **kwargs):

        # super().__init__(policy, env, verbose=verbose, _init_setup_model=False, **kwargs)
        logger.log("Init")
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

        # GAIL Params
        self.use_gasil = use_gasil
        self.g_step = g_step  # TODO: use this

        self.sil_alpha = sil_alpha
        self.sil_samples = sil_samples
        self.buffer = RewardBuffer(size=sil_samples)

        self.adversary = None
        self.adversary_step = adversary_step
        self.adversary_stepsize = adversary_stepsize
        self.adversary_entcoeff = adversary_entcoeff
        self.adversary_hidden_size = adversary_hidden_size

        # For HParam Logging
        self.env_name = env_name


        # Keep Track of Episode Reward (set, update, log from Callback)
        self.real_reward = []  # Env Reward Buffer for Adapted Rollouts
        self.d_reward = []     # Discriminator Reward Buffer for Adapted Rollouts
        self.accumulated_env_reward = np.zeros((self.n_envs,))  # []

        if _init_setup_model:  # TODO: revert changes to PPO model setup
            self.setup_model()

        with self.graph.as_default():
            # self.set_random_seed(self.seed)
            # self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

            self.adversary = TransitionClassifier(self.observation_space, self.action_space,
                                                     self.adversary_hidden_size, self.sess,
                                                     self.train_model.obs_ph, self.action_ph,
                                                     stepsize=self.adversary_stepsize,
                                                     entcoeff=self.adversary_entcoeff)
            # tf_util.initialize(sess=self.sess)

            self.params.extend(self.adversary.get_trainable_variables())
            tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101
            self.summary = tf.summary.merge_all()  # TODO test if needed / usable

            self.adversary.setup_trainer()

            # with tf.variable_scope("DiscriminatorAdam", reuse=False): # TODO: Move into Adversary
            #     self.d_adam = MpiAdam(self.adversary.get_trainable_variables(), sess=self.sess)
            #     self.d_adam.sync()

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update,
                    writer, states=None, cliprange_vf=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        :param cliprange_vf: (float) Clipping factor for the value function
        """
        # logger.log("Train Step from GASIL")
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions,
                  self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        expert_obs, expert_acs, _ = self.buffer.sample(len(obs))
        td_map[self.adversary.expert_obs_ph] = expert_obs
        td_map[self.adversary.expert_acs_ph] = expert_acs

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.clip_range_vf_ph] = cliprange_vf

        if states is None:
            update_fac = max(self.n_batch // self.nminibatches // self.noptepochs, 1)
        else:
            update_fac = max(self.n_batch // self.nminibatches // self.noptepochs // self.n_steps, 1)

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train], td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="GASIL",
              reset_num_timesteps=True):
        callback = LeanAdversaryCallback(verbose=self.verbose)
        return super().learn(
            total_timesteps, callback=callback, log_interval=log_interval,
            tb_log_name='', reset_num_timesteps=reset_num_timesteps)

    def write_hparams(self, writer):
        """Returns a summary proto buffer holding this experiment"""
        import hashlib
        from tensorboard.plugins.hparams import api_pb2
        from tensorboard.plugins.hparams import summary

        hparams = {
            'alpha': self.sil_alpha,
            'buffer': self.sil_samples
        }
        # "GASIL Evaluation"
        group_name = hashlib.md5(str(hparams).encode('utf-8')).hexdigest()
        group_name = "{}:{}_{}".format(self.env_name, self.sil_samples, self.sil_alpha)

        writer.add_summary(summary.experiment_pb(
            hparam_infos=[
                api_pb2.HParamInfo(
                    name="alpha", display_name="Reward Mixture", type=api_pb2.DATA_TYPE_FLOAT64,
                    domain_interval=api_pb2.Interval(min_value=0.0, max_value=1.0)),
                api_pb2.HParamInfo(
                    name="buffer", display_name="Imitation Buffer Size", type=api_pb2.DATA_TYPE_FLOAT64,
                    domain_interval=api_pb2.Interval(min_value=64.0, max_value=1024.0))
            ],
            metric_infos=[
                api_pb2.MetricInfo(
                    name=api_pb2.MetricName(tag="rewards/true_acc_reward"),
                    display_name="Accumulated Env Reward"),
                api_pb2.MetricInfo(
                    name=api_pb2.MetricName(tag="adversary_loss/mean_reward_in_buffer"),
                    display_name="Mean Reward in Buffer")
            ]
        ))
        writer.add_summary(summary.session_start_pb(hparams=hparams, group_name=group_name))
        writer.flush()

    def finish_hparams(self, writer):
        """Returns a summary proto buffer holding this experiment"""
        from tensorboard.plugins.hparams import api_pb2
        from tensorboard.plugins.hparams import summary
        writer.add_summary(summary.session_end_pb(api_pb2.STATUS_SUCCESS))
        writer.flush()

    def save(self, save_path, cloudpickle=False):
        # verbose=0, tensorboard_log=None, _init_setup_model=True, full_tensorboard_log=False,
        data = {
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
            "adversary_hidden_size": self.adversary_hidden_size,
            "adversary_entcoeff": self.adversary_entcoeff,
            "g_step": self.g_step,
            "adversary_step": self.adversary_step,
            "adversary_stepsize": self.adversary_stepsize,
            "sil_samples": self.sil_samples,
            "use_gasil": self.use_gasil
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)


class LeanAdversaryCallback(BaseCallback):
    """
    A custom callback Updating the Generative Adversarial Self-Imitation Extension

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(LeanAdversaryCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        w = self.locals['writer']
        self.model.write_hparams(writer=w)

        # HP_ALPHA = hp.HParam('alpha', hp.RealInterval(0.0, 1.0))
        # HP_BUFFER = hp.HParam("buffer", hp.IntInterval(64, 1024))
        # # HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
        # HP_METRIC = hp.Metric('rewards/true_acc_reward', display_name='Reward')
        #
        # # self.model.sess.run(w.init())
        # self.model.sess.run(hp.hparams_config(
        #     hparams=[HP_ALPHA, HP_BUFFER], metrics=[HP_METRIC])
        # )
        # self.model.sess.run(hp.hparams({
        #   HP_ALPHA: self.model.sil_alpha,
        #   HP_BUFFER: self.model.sil_samples
        # }))
        # self.model.sess.run(w.flush())

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # self.locals['d_rewards'] = []

        # Reset Env Reward Tracker
        # self.model.real_reward = []
        self.model.d_rewards = []

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        observation = self.locals['mb_obs'][-1].copy()
        action = self.locals['clipped_actions']
        reward = self.model.adversary.get_reward(observation, action[0])
        self.model.d_rewards.append(reward[0])
        env_reward = self.locals['rewards']

        # Write Detailed Reward Summary
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='rewards/discriminator', simple_value=reward),
            tf.Summary.Value(tag='rewards/environment', simple_value=env_reward[0])
        ])
        self.locals['writer'].add_summary(summary, self.num_timesteps)

        # For discounting D rewards after Rollout

        if self.num_timesteps % self.model.n_steps == 0:

            # Reshabe Discriminator Reward
            d_rewards = np.asarray(self.model.d_rewards, dtype=np.float32)
            mb_rewards = self.locals['mb_rewards'][:]
            mb_rewards.append(self.locals['rewards'])
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_values = np.asarray(self.locals['mb_values'], dtype=np.float32)
            mb_dones = np.asarray(self.locals['mb_dones'], dtype=np.bool)

            obs = self.locals['obs']
            states = self.locals['states']
            dones = self.locals['dones']
            last_values = self.model.value(obs, states, dones)

            # discount/bootstrap off value fn
            # last_values = self.model.model.value(self.model.obs, self.model.states, self.model.dones)
            d_adv = np.zeros_like(d_rewards)
            d_true_reward = np.copy(d_rewards)
            last_gae_lam = 0
            for step in reversed(range(self.model.n_steps)):
                if step == self.model.n_steps - 1:
                    nextnonterminal = 1.0 - dones  # self.model.dones
                    nextvalues = last_values
                else:
                    nextnonterminal = 1.0 - mb_dones[step + 1]  # self.d_mb_dones[step + 1]
                    nextvalues = mb_values[step + 1]    # self.d_mb_values[step + 1]
                mixted_reward = self.model.sil_alpha * d_rewards[step] + (1-self.model.sil_alpha) * mb_rewards[step]
                delta = mixted_reward + self.model.gamma * nextvalues * nextnonterminal - mb_values[step]  # self.d_mb_values[step]
                d_adv[step] = last_gae_lam = delta + self.model.gamma * self.model.lam * nextnonterminal * last_gae_lam
            d_returns = d_adv + mb_values

            # Write Value to reward summary
            self.locals['writer'].add_summary(tf.Summary(value=[tf.Summary.Value(
                tag='rewards/value', simple_value=np.mean(mb_values, axis=1)[0])
            ]), self.num_timesteps)

            d_returns, d_true_reward = map(swap_and_flatten, (d_returns, d_true_reward))
            self.d_returns = d_returns
            self.d_true_reward = d_true_reward

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # logger.log("Rollout Ended. Updating Policy and Discriminator")

        # Get Variables
        observations = self.locals['obs']
        actions = self.locals['actions']
        returns = self.locals['returns']

        true_rewards = self.locals['true_reward']
        masks = self.locals['masks']
        n_envs = self.model.n_envs  # self.locals['n_envs']
        n_steps = self.model.n_steps  # self.locals['n_steps']
        writer = self.locals['writer']
        steps = self.num_timesteps

        rewards = true_rewards.reshape((n_envs, n_steps))
        masks = masks.reshape((n_envs, n_steps))

        # Write Real Env Reward to Tensoboard
        self.model.accumulated_env_reward = total_episode_reward_logger(
            self.model.accumulated_env_reward,
            rewards, masks, writer, steps,
            tag="rewards/true_acc_reward"
        )

        # Update Buffer
        self.model.buffer.extend(observations, actions, returns)
        buffer_rewards = [reward for _, _, reward in self.model.buffer.storage]

        assert len(observations) == self.model.n_steps
        batch_size = self.model.n_steps // self.model.adversary_step
        d_losses = self.model.adversary.learn(
            observations, actions, self.model.buffer, batch_size
        )
        logger.logkv("d_losses", np.mean(d_losses, axis=1)[0])

        # Write Detailed Reward Summary
        writer.add_summary(tf.Summary(value=[tf.Summary.Value(
            tag='adversary_loss/discriminator_loss',
            simple_value=np.mean(d_losses, axis=1)[0])
        ]), self.num_timesteps)

        # Overwrite PPO Return & Reward
        if self.model.use_gasil:
            self.locals['true_reward'] = self.d_true_reward  # d_rewards
            self.locals['returns'] = self.d_returns  # d_rewards

        # Write Tensorboard Summary
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='adversary_loss/samples_in_buffer',
                             simple_value=len(self.model.buffer)),
            tf.Summary.Value(tag='adversary_loss/mean_reward_in_buffer',
                             simple_value=np.mean(buffer_rewards))
        ])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
