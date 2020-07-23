"""
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
Taken from /gail/adversary, adapted to Self-Imitation Needs
-> Added Reward to input
"""
import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.common import tf_util, dataset
from stable_baselines.common.mpi_adam import MpiAdam


def logsigmoid(input_tensor):
    """
    Equivalent to tf.log(tf.sigmoid(a))

    :param input_tensor: (tf.Tensor)
    :return: (tf.Tensor)
    """
    return -tf.nn.softplus(-input_tensor)


def logit_bernoulli_entropy(logits):
    """
    Reference:
    https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51

    :param logits: (tf.Tensor) the logits
    :return: (tf.Tensor) the Bernoulli entropy
    """
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


class TransitionClassifier(object):
    def __init__(self, observation_space, action_space, hidden_size, session,
                 generator_obs_ph, generator_acs_ph, stepsize=3e-4,
                 entcoeff=0.001, scope="adversary", normalize=True):
        """
        Reward regression from observations and transitions

        :param observation_space: (gym.spaces)
        :param action_space: (gym.spaces)
        :param hidden_size: ([int]) the hidden dimension for the MLP
        :param session: (tf.Session) the current TensorFlow session
        :param generator_obs_ph: (tf.Placeholder) Generator(Policy) Observation Placeholders
        :param generator_acs_ph: (tf.Placeholder) Generator(Policy) Action Placeholders
        :param stepsize: (float) the discriminator stepsize
        :param entcoeff: (float) the entropy loss weight
        :param scope: (str) tensorflow variable scope
        :param normalize: (bool) Whether to normalize the reward or not
        """
        # TODO: support images properly (using a CNN)
        self.scope = scope
        self.action_space = action_space
        self.observation_shape = observation_space.shape
        self.actions_shape = action_space.shape
        self.session = session
        self.trainer = None

        if isinstance(action_space, gym.spaces.Box):
            # Continuous action space
            self.discrete_actions = False
            self.n_actions = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.Discrete):
            self.n_actions = action_space.n
            self.discrete_actions = True
        else:
            raise ValueError('Action space not supported: {}'.format(action_space))

        self.hidden_size = hidden_size
        self.normalize = normalize
        self.stepsize = stepsize
        self.obs_rms = None

        # with tf.variable_scope("adversary_model", reuse=True):
            # Placeholders
        self.generator_obs_ph = generator_obs_ph
        self.generator_acs_ph = generator_acs_ph

        self.generator_rew_ph = tf.placeholder(tf.float32, (None, 1), name="generator_reward_ph")

        # self.generator_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
        #                                        name="observations_ph")
        # self.generator_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
        #                                        name="actions_ph")
        self.expert_obs_ph = tf.placeholder(observation_space.dtype, (None,) + self.observation_shape,
                                            name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                            name="expert_actions_ph")
        self.expert_rew_ph = tf.placeholder(tf.float32, (None, 1), name="expert_reward_ph")

        # Build graph
        generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, self.generator_rew_ph, reuse=False)
        expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, self.expert_rew_ph, reuse=True)

        with tf.variable_scope("adversary_loss", reuse=False):
            # Build accuracy
            generator_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(generator_logits) < 0.5, tf.float32))
            expert_acc = tf.reduce_mean(tf.cast(tf.nn.sigmoid(expert_logits) > 0.5, tf.float32))

            # Build regression loss
            # let x = logits, z = targets.
            # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                     labels=tf.zeros_like(generator_logits))
            generator_loss = tf.reduce_mean(generator_loss)
            expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
            expert_loss = tf.reduce_mean(expert_loss)
            # Build entropy loss
            logits = tf.concat([generator_logits, expert_logits], 0)
            entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
            entropy_loss = -entcoeff * entropy
            # Loss + Accuracy terms
            self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
            self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
            self.total_loss = generator_loss + expert_loss + entropy_loss
            # Build Reward for policy
            self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
            var_list = self.get_trainable_variables()
            self.lossandgrad = tf_util.function(
                [self.generator_obs_ph, self.generator_acs_ph, self.generator_rew_ph,
                    self.expert_obs_ph, self.expert_acs_ph, self.expert_rew_ph],
                self.losses + [tf_util.flatgrad(self.total_loss, var_list)])

            tf.summary.scalar('generator_loss', generator_loss)
            tf.summary.scalar('expert_loss', expert_loss)
            tf.summary.scalar('entropy', entropy)
            tf.summary.scalar('generator_acc', generator_acc)
            tf.summary.scalar('expert_acc', expert_acc)

    def build_graph(self, obs_ph, acs_ph, rew_ph, reuse=False):
        """
        build the graph

        :param obs_ph: (tf.Tensor) the observation placeholder
        :param acs_ph: (tf.Tensor) the action placeholder
        :param reuse: (bool)
        :return: (tf.Tensor) the graph output
        """
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if self.normalize:
                with tf.variable_scope("obfilter"):
                    self.obs_rms = RunningMeanStd(shape=self.observation_shape)
                obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            else:
                obs = obs_ph

            if self.discrete_actions:
                one_hot_actions = tf.one_hot(acs_ph, self.n_actions)
                actions_ph = tf.cast(one_hot_actions, tf.float32)
            else:
                actions_ph = acs_ph

            _input = tf.concat([obs, actions_ph, rew_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        """
        Get all the trainable variables from the graph

        :return: ([tf.Tensor]) the variables
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def setup_trainer(self):
        with tf.variable_scope("adversary_trainer", reuse=False):
            self.trainer = MpiAdam(
                self.get_trainable_variables(), sess=self.session
            )
            self.trainer.sync()

    def get_reward(self, obs, actions, reward):
        """
        Predict the reward using the observation and action

        :param obs: (tf.Tensor or np.ndarray) the observation
        :param actions: (tf.Tensor or np.ndarray) the action
        :param reward: (tf.Tensor or np.ndarray) the environment reward
        :return: (np.ndarray) the discriminator reward
        """
        # sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 0)
        elif len(actions.shape) == 0:
            # one discrete action
            actions = np.expand_dims(actions, 0)

        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: actions, self.generator_rew_ph: reward}
        reward = self.session.run(self.reward_op, feed_dict)
        return reward

    def learn(self, observations, actions, rewards, buffer, batch_size):
        """
        Update the discriminator network with current observations and buffer
        :param observations: (np.ndarray) the last policy observations
        :param actions: (np.ndarray) the last policy actions
        :param rewards: (np.ndarray) the last environment rewards
        :param buffer: (gasil.buffer) the current Self-Imitation Buffer
        :param batch_size: (int) the number of samples for each update iteration
        :return: ([float]) a list of losses
        """
        with self.session.as_default():
            # ------------------ Update D ------------------
            # logger.log("Optimizing Discriminator...")

            # NOTE: uses only the last g step for observation
            d_losses = []  # list of tuples, each of which gives the loss for a minibatch
            # NOTE: for recurrent policies, use shuffle=False?
            for ob_batch, ac_batch, re_batch in dataset.iterbatches(
                (observations, actions, rewards), batch_size=batch_size,
                    include_final_partial_batch=False, shuffle=True):

                ob_expert, ac_expert, re_expert, _ = buffer.sample(batch_size=batch_size)
                # ob_expert, ac_expert = self.expert_dataset.get_next_batch()
                # update running mean/std for reward_giver
                # TODO whats that doing ??
                if self.normalize:
                    self.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))

                # Reshape actions if needed when using discrete actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    if len(ac_batch.shape) == 2:
                        ac_batch = ac_batch[:, 0]
                    if len(ac_expert.shape) == 2:
                        ac_expert = ac_expert[:, 0]
                *newlosses, grad = self.lossandgrad(
                    ob_batch, ac_batch, re_batch,
                    ob_expert, ac_expert, re_expert)  # , sess=self.model.sess
                # Allmean from TRPO not needed ? using MPI self.allmean(grad)
                self.trainer.update(grad, self.stepsize)
                d_losses.append(newlosses)
        return d_losses
