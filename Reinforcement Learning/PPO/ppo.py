from collections import deque
import random
import gym
from gym import wrappers
import numpy as np
import tensorflow as tf
import datetime
from pathlib import Path, PurePath
import os
import argparse
from scipy import signal
from statistics import mean

# This class comes from "https://github.com/openai/spinningup/blob/2e0eff9bd019c317af908b72c056a33f14626602/spinup/algos/ppo/ppo.py"
def discount_cumsum(x, discount):
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

# This class comes from "https://github.com/openai/spinningup/blob/2e0eff9bd019c317af908b72c056a33f14626602/spinup/algos/ppo/ppo.py"
class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.int32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        # used_size = self.ptr
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)           # only single thread for now, so we don't need MPI
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]
        # return [self.obs_buf[:used_size], self.act_buf[:used_size], self.adv_buf[:used_size], self.ret_buf[:used_size], self.logp_buf[:used_size]]


class PPO(object):
    # def __init__(self, state_dim, action_dim, lr, exp_dir=None, max_norm=10.0, soft_update=True, entropy_coef=0.01, tau=0.001, init_buffer_size=1000):
    def __init__(self, state_dim, action_dim, lr, exp_dir, train_epochs=5, clip_ratio=0.2, steps_per_epoch=2000, gamma=0.99, lam=0.95, max_norm=None, entropy_coef=0.0, save_freq=10):
        # self._epoch = 0
        self._save_freq = save_freq
        self._buffer = PPOBuffer(state_dim, action_dim, steps_per_epoch, gamma, lam)
        self._lr = lr
        self._clip_ratio = clip_ratio
        self.train_epochs = train_epochs
        self._actor_net = self._create_actor(state_dim, action_dim)
        self._critic_net = self._create_critic(state_dim, action_dim)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._optimizer = tf.keras.optimizers.Adam(lr)
        self._max_norm = max_norm
        self._entropy_coef = entropy_coef
        self._exp_dir = exp_dir

        os.makedirs(str(self._exp_dir), exist_ok=True)
        self._log_dir = self._exp_dir / "tensorboard"
        self._checkpoint_dir = self._exp_dir / "checkpoints"

        self.checkpoint = tf.train.Checkpoint(epoch=tf.Variable(1), actor_net=self._actor_net, critic_net=self._critic_net, optimizer=self._optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, str(self._checkpoint_dir), max_to_keep=None)
        self._find_existing_checkpoints()
        self._create_metrics()

    def restore(self, checkpoint_path):
        self.checkpoint.restore(checkpoint_path)
        print("Restored from {}".format(checkpoint_path))

    def _create_metrics(self):
        self.summary_writer = tf.summary.create_file_writer(str(self._log_dir))
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.td_loss = tf.keras.metrics.Mean('td_loss', dtype=tf.float32)
        self.entropy = tf.keras.metrics.Mean('entropy', dtype=tf.float32)

    def _find_existing_checkpoints(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print("Restored from {}".format(self.checkpoint_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    def _create_actor(self, state_dim, action_dim):
        inputs = tf.keras.Input(shape=state_dim)
        fc1 = tf.keras.layers.Dense(512, activation='relu')(inputs)
        fc2 = tf.keras.layers.Dense(256, activation='relu')(fc1)
        pi_outs = tf.keras.layers.Dense(action_dim)(fc2)

        model = tf.keras.Model(inputs=inputs, outputs=pi_outs)
        return model

    def _create_critic(self, state_dim, action_dim):
        inputs = tf.keras.Input(shape=state_dim)
        fc1 = tf.keras.layers.Dense(512, activation='relu')(inputs)
        fc2 = tf.keras.layers.Dense(256, activation='relu')(fc1)
        v_outs = tf.keras.layers.Dense(1)(fc2)

        model = tf.keras.Model(inputs=inputs, outputs=v_outs)
        return model

    def act(self, state):
        pi_logits = self._actor_net(np.expand_dims(state,0))
        v = self._critic_net(np.expand_dims(state,0))

        '''
        if self.step % 100 == 0:
            with self.summary_writer.as_default():
                tf.summary.histogram("q_logits", act_dist, step=self.step)
                tf.summary.histogram("act_dist", tf.nn.softmax(act_dist), step=self.step)
        '''
        
        a = tf.random.categorical(pi_logits, num_samples=1)
        log_pi_logits = tf.nn.log_softmax(pi_logits)
        logp_a = tf.gather_nd(log_pi_logits, indices=a, batch_dims=1)

        return a.numpy()[0][0], v, logp_a

    def get_pi_logp(self, states, actions):
        actions = np.expand_dims(actions, -1)
        pi_logits = self._actor_net(states)
        log_pi_logits = tf.nn.log_softmax(pi_logits)
        selected_logp = tf.gather_nd(log_pi_logits, indices=actions, batch_dims=1)
        return pi_logits, selected_logp

    def get_val(self, state):
        v = self._critic_net(np.expand_dims(state, 0))
        return v

    def save_model(self):
        saved_path = self.checkpoint_manager.save(self.checkpoint.epoch)
        print('model saved at {}'.format(saved_path))

    def save_transition(self, s, a, r, v, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self._buffer.store(s, a, r, v, logp)

    def finish_path(self, last_val):
        self._buffer.finish_path(last_val)

    def train(self):
        s_batch, a_batch, adv_batch, return_batch, logp_old_batch = self._buffer.get()

        for train_epoch in range(self.train_epochs):
            with tf.GradientTape() as tape:
                # Clipped Objective
                pi_logits, logp_batch = self.get_pi_logp(s_batch, a_batch)
                importance_ratio = tf.exp(logp_batch - logp_old_batch)
                importance_ratio_clipped = tf.clip_by_value(importance_ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
                objective = importance_ratio * adv_batch
                objective_clipped = importance_ratio_clipped * adv_batch
                pi_loss = -tf.reduce_mean(tf.minimum(objective, objective_clipped))

                # Entropy
                entropy_reg = tf.reduce_sum(-tf.nn.softmax(pi_logits) * tf.nn.log_softmax(pi_logits), -1)

                # Value Loss
                v_batch = self.get_val(s_batch)
                v_loss = tf.reduce_mean(tf.math.squared_difference(return_batch, v_batch))

                # Total Loss
                loss = pi_loss + v_loss + self._entropy_coef * entropy_reg

            # Gradient Clipping
            variables_to_train = (self._actor_net.trainable_variables + self._critic_net.trainable_variables)
            if self._max_norm != None:
                gradients = tape.gradient(loss, variables_to_train)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, self._max_norm)
                self._optimizer.apply_gradients(zip(clipped_gradients, variables_to_train))
            else:
                gradients = tape.gradient(loss, variables_to_train)
                self._optimizer.apply_gradients(zip(gradients, variables_to_train))


            # Stop optimizing if kl-divergence is too large
            _, logp_new_batch = self.get_pi_logp(s_batch, a_batch)
            sampled_kl = tf.reduce_mean(logp_old_batch - logp_new_batch)      # a sample estimate for KL-divergence, easy to compute
            if sampled_kl > 0.05 * 1.5:
                print('Early stopping at traning epoch {} due to reaching max kl.'.format(train_epoch))
                break

        # self.entropy(entropy_reg)
        # self.td_loss(td_loss)
        # self.train_loss(loss)

        # Save Model
        if self.checkpoint.epoch % self._save_freq == 0:
            self.save_model()
        
        self.checkpoint.epoch.assign_add(1)

        '''
        # Write log to tensorboard
        if self.step % 100 == 0:
            with self.summary_writer.as_default():^
                tf.summary.scalar("train_loss", self.train_loss.result(), step=self.step)
                tf.summary.scalar("td_loss", self.td_loss.result(), step=self.step)
                tf.summary.scalar("entropy", self.entropy.result(), step=self.step)
        '''
                
def main(args):
    if not args.evaluate:
        train(args)
    else:
        eval(args)

def train(args):
    # Create experiement dir
    if args.checkpoint_path is None:
        now = datetime.datetime.now()
        exp_dir = Path(now.strftime("PPO_{}_%m_%d_%Y-%H_%M_%S").format(args.environment))
    else:
        exp_dir = Path(args.checkpoint_path)

    steps_per_epoch = args.steps_per_epoch

    env = gym.make(args.environment)
    agent = PPO(state_dim=env.observation_space.shape,
                action_dim=env.action_space.n,
                lr=args.learning_rate,
                exp_dir=exp_dir,
                steps_per_epoch=steps_per_epoch)
    
    max_score = 0
    e_count = 0
    s, r, done, ep_score = env.reset(), 0, False, 0
    # epoch_count = 1

    while True:
        # Steps per epoch
        for t in range(steps_per_epoch):
            a, v, logp = agent.act(s)
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            # So we save the transitions before apply the actions
            agent.save_transition(s, a, r, v, logp)

            s, r, done, _ = env.step(a)
            ep_score += r
            # step_count += 1

            # if the episode is terminated or reach the maximum steps
            if done or t == steps_per_epoch - 1:
                # e_count += 1
                last_val = r if done else agent.get_val(s)
                agent.finish_path(last_val)

                # reset the environment if the episode is finished
                if done:
                    e_count += 1
                    max_score = max(max_score, ep_score)
                    print('episode:', e_count, 'score:', ep_score, 'max:', max_score)
                    s, r, done, ep_score = env.reset(), 0, False, 0            # reset the environment

        agent.train()        

        '''
        # Evaluation
        score_list = []
        for _ in range(5):
            s = env.reset()
            score = 0
            while True:
                a = agent.act(s)
                next_s, reward, done, _ = env.step(a)
                score += reward
                s = next_s
                if done:
                    score_list.append(score)
                    break
        
        with agent.summary_writer.as_default():
            tf.summary.histogram("Evaluation Scores", score_list, step=agent.step)
        '''
        
    env.close()


def eval(args):
    if args.checkpoint_path == None:
        print("Please provide the path of checkpoint you want to evaluate")
        return

    env = gym.make(args.environment)
    # env = wrappers.Monitor(env, './eval_videos/{}/'.format(datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S")))
    agent = PPO(state_dim=env.observation_space.shape, action_dim=env.action_space.n, lr=args.learning_rate)
    agent.restore(args.checkpoint_path)

    eval_episodes = 10
    score_list = []
    for _ in range(eval_episodes):
        s = env.reset()
        score = 0
        while True:
            a = agent.act(s)
            env.render()
            next_s, reward, done, _ = env.step(a)
            score += reward
            s = next_s
            if done:
                score_list.append(score)
                break

    print ("Evaluated the agent for {} episodes.".format(eval_episodes))
    print ("Max: {}".format(max(score_list)))
    print ("Mean: {}".format(mean(score_list)))
    print ("Min: {}".format(min(score_list)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', '--environment', help='The name of OpenAI Gym Envrionment. e.g. CartPole-v1', type=str, default="CartPole-v1")
    parser.add_argument('-spe', '--steps_per_epoch', help="The number of steps to collect for one epoch", type=int, default=200)
    parser.add_argument('-lr', '--learning_rate', help='Learning Rate', type=float, default=0.0005)
    parser.add_argument('-eval', '--evaluate', help="Evaluate mode", action="store_true")
    parser.add_argument('-cp', '--checkpoint_path', help="Checkpoint path", type=str, default=None)
    args = parser.parse_args()
    main(args)