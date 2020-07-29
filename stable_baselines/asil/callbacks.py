import numpy as np
import tensorflow as tf
from stable_baselines import logger
from stable_baselines.common.callbacks import BaseCallback


class LoggingCallback(BaseCallback):
    """
    A custom callback Updating the Generative Adversarial Self-Imitation Extension

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, env_name="", verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.env_name = env_name
        logger.log("Init LoggingCallback {}".format(env_name))

    def make_hparams(self):
        from tensorboard.plugins.hparams import api_pb2
        hparams = {
            'alpha': self.model.sil_alpha,
            'update': self.model.sil_update,
            'buffer': self.model.sil_samples
        }
        group_name = "{}:{}_{}".format(self.env_name, hparams['buffer'], hparams['alpha'])
        hparam_infos = [
            api_pb2.HParamInfo(
                name="alpha", display_name="Reward Mixture", type=api_pb2.DATA_TYPE_FLOAT64,
                domain_interval=api_pb2.Interval(min_value=0.0, max_value=1.0)),
            api_pb2.HParamInfo(
                name="update", display_name="Adversary Update Frequency", type=api_pb2.DATA_TYPE_FLOAT64,
                domain_interval=api_pb2.Interval(min_value=0.0, max_value=1.0)),
            api_pb2.HParamInfo(
                name="buffer", display_name="Imitation Buffer Size", type=api_pb2.DATA_TYPE_FLOAT64,
                domain_interval=api_pb2.Interval(min_value=64.0, max_value=1024.0))
        ]
        metric_infos = [
            api_pb2.MetricInfo(
                name=api_pb2.MetricName(tag="rewards/environment_reward"),
                display_name="Accumulated Environment Reward"),
            api_pb2.MetricInfo(
                name=api_pb2.MetricName(tag="adversary_loss/mean_reward_in_buffer"),
                display_name="Mean Reward in Buffer"),
            api_pb2.MetricInfo(
                name=api_pb2.MetricName(tag="rewards/mean_reward"),
                display_name="Mean Reward over last 100 Episodes"),
            api_pb2.MetricInfo(
                name=api_pb2.MetricName(tag="rewards/number_episodes"),
                display_name="Number of training episodes")
        ]
        return hparams, group_name, hparam_infos, metric_infos

    def write_hparams(self, writer, hparams, group_name, hparam_infos=[], metric_infos=[]):
        """Returns a summary proto buffer holding this experiment"""
        from tensorboard.plugins.hparams import api_pb2
        from tensorboard.plugins.hparams import summary

        writer.add_summary(summary.experiment_pb(
            hparam_infos=hparam_infos, metric_infos=metric_infos)
        )
        writer.add_summary(summary.session_start_pb(
            hparams=hparams, group_name=group_name)
        )
        writer.flush()

    def finish_hparams(self, writer, success=False):
        """Returns a summary proto buffer holding this experiment"""
        from tensorboard.plugins.hparams import api_pb2
        from tensorboard.plugins.hparams import summary
        message = api_pb2.STATUS_SUCCESS if success else api_pb2.STATUS_FAILURE
        writer.add_summary(summary.session_end_pb(message))
        writer.flush()

    def _on_training_start(self):
        from tensorboard.plugins.text import metadata

        hparams, group_name, hparam_infos, metric_infos = self.make_hparams()
        writer = self.locals['writer']
        self.write_hparams(writer, hparams, group_name, hparam_infos, metric_infos)

        params = self.model.get_hparams()
        table = [[k, str(v)] for k, v in params.items()]
        metadata = metadata.create_summary_metadata(
                display_name="HParams", description="Hyper Parameters")
        metadata = tf.SummaryMetadata.FromString(metadata.SerializeToString())
        tensor = tf.make_tensor_proto(table, dtype=tf.string)
        writer.add_summary(tf.Summary(value=[
            tf.Summary.Value(tag="HParams", metadata=metadata, tensor=tensor)]
        ), 0)
        writer.flush()

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # Get Variables
        # returns = self.locals['returns']
        values = self.locals['values']
        reward = self.model.d_rewards
        true_reward = self.locals['true_reward']
        ep_infos = self.locals['ep_infos']
        writer = self.locals['writer']
        buffer_rewards = [r for _, _, r in self.model.buffer.storage]

        # Write Success to HParams
        if self.model.stop:
            success = self.model.total_episodes < self.model.max_episodes
            self.finish_hparams(writer, success=success)
            return

        for info in ep_infos:
            self.model.total_episodes += 1
            self.model.current_episode += info['l']
            self.model.moving_reward.append(info['r'])
            writer.add_summary(tf.Summary(value=[
                tf.Summary.Value(tag='rewards/episode_length',
                                 simple_value=info['l']),
                tf.Summary.Value(tag='rewards/environment_reward',
                                 simple_value=info['r']),
                tf.Summary.Value(tag='rewards/mean_reward',
                                 simple_value=np.mean(self.model.moving_reward)),
                tf.Summary.Value(tag='rewards/number_episodes',
                                 simple_value=self.model.total_episodes)
            ]), self.model.current_episode)

        # Write Detailed Reward Summary
        writer.add_summary(tf.Summary(value=[
            tf.Summary.Value(tag='rewards/discriminator',
                             simple_value=np.mean(reward)),
            tf.Summary.Value(tag='rewards/environment',
                             simple_value=np.mean(true_reward)),
            tf.Summary.Value(tag='rewards/value',
                             simple_value=np.mean(values)),
            tf.Summary.Value(tag='adversary_loss/samples_in_buffer',
                             simple_value=len(self.model.buffer)),
            tf.Summary.Value(tag='adversary_loss/mean_reward_in_buffer',
                             simple_value=np.mean(buffer_rewards)),
            tf.Summary.Value(tag='adversary_loss/buffer_updates',
                             simple_value=self.model.buffer.overwrites)
        ]), self.num_timesteps)
        logger.logkv("100_mean_reward", np.mean(self.model.moving_reward))


class EarlyStoppingCallback(BaseCallback):
    """
    A custom callback terminating Learning when a threshold is reached
    :param threshold: (int) Solving threshold of 100-episode mean reward
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, threshold, env_name, verbose=0):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.threshold = threshold
        self.env_name = env_name
        logger.log("Init EarlyStoppingCallback {}".format(threshold))

    def _on_step(self):
        if not self.model.terminate_on_solve:
            return True
        if np.mean(self.model.moving_reward) >= self.threshold:
            self.logger.log("Solved {} in {} Episodes".format(
                self.env_name, self.model.total_episodes))
            self.model.stop = True
            return False
        if self.model.total_episodes > self.model.max_episodes:
            self.logger.log("Did not solve {}".format(self.env_name))
            self.model.stop = True
            return False
