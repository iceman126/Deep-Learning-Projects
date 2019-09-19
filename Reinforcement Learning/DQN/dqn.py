from collections import deque
import random
import gym
import numpy as np
import tensorflow as tf
import datetime
from pathlib import Path, PurePath
import os
import argparse
from statistics import mean

class DQN(object):
    def __init__(self, state_dim, action_dim, lr, exp_dir=None, max_norm=10.0, soft_update=True, entropy_coef=0.01, tau=0.001, init_buffer_size=1000):
        self.step = 0
        self.save_freq = 1000
        self.update_freq = 1000
        self.replay_buffer_size = 500000
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.lr = lr
        self.model = self._create_model(state_dim, action_dim)
        self.target_model = self._create_model(state_dim, action_dim)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._init_buffer_size = init_buffer_size
        # self.optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.target_model.set_weights(self.model.get_weights())
        self._soft_update = soft_update
        self._tau = tau
        self._max_norm = max_norm
        self._entropy_coef = entropy_coef
        
        if exp_dir == None:
            now = datetime.datetime.now()
            self.exp_dir = Path(now.strftime("%m_%d_%Y-%H_%M_%S"))
            try:
                os.makedirs(str(self.exp_dir))
            except FileExistsError:
                print("Directory already exists")
        else:
            self.exp_dir = Path(exp_dir)

        self.log_dir = self.exp_dir / "tensorboard"
        self.checkpoint_dir = self.exp_dir / "checkpoints"

        self.checkpoint = tf.train.Checkpoint(epoch=tf.Variable(1), q_net=self.model, target_q_net=self.target_model, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, str(self.checkpoint_dir), max_to_keep=None)
        self._find_existing_checkpoints()
        self._create_metrics()

    def _run_soft_update(self):
        for (b_tensor, t_tensor) in zip(self.model.trainable_variables, self.target_model.trainable_variables):
            t_tensor.assign((1 - self._tau) * t_tensor + self._tau * b_tensor)

    def _create_metrics(self):
        self.summary_writer = tf.summary.create_file_writer(str(self.log_dir))
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.td_loss = tf.keras.metrics.Mean('td_loss', dtype=tf.float32)
        self.entropy = tf.keras.metrics.Mean('entropy', dtype=tf.float32)

    def _find_existing_checkpoints(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print("Restored from {}".format(self.checkpoint_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    def _create_model(self, state_dim, action_dim):
        inputs = tf.keras.Input(shape=state_dim)
        fc1 = tf.keras.layers.Dense(512, activation='relu')(inputs)
        fc2 = tf.keras.layers.Dense(256, activation='relu')(fc1)
        q_outs = tf.keras.layers.Dense(action_dim)(fc2)

        model = tf.keras.Model(inputs=inputs, outputs=q_outs)
        return model

    def act(self, state):
        act_dist = self.model(np.expand_dims(state,0))

        if self.step % 100 == 0:
            with self.summary_writer.as_default():
                tf.summary.histogram("q_logits", act_dist, step=self.step)
                tf.summary.histogram("act_dist", tf.nn.softmax(act_dist), step=self.step)
        
        sampled_action = tf.random.categorical(act_dist, num_samples=1).numpy()[0][0]
        return sampled_action

    def save_model(self):
        saved_path = self.checkpoint_manager.save(self.step)
        print('model saved at {}'.format(saved_path))

    def save_transition(self, s, a, r, next_s, terminated):
        self.replay_buffer.append((s, a, r, next_s, int(terminated == False)))

    def train(self, batch_size=512, lr=1, gamma=0.99):
        # TODO: Need to find a better place to increase this variable
        self.step += 1

        if len(self.replay_buffer) < self._init_buffer_size:
            return
        
        minibatch = random.sample(self.replay_buffer, batch_size)
        s_batch = np.array([replay[0] for replay in minibatch])
        selected_actions = np.expand_dims(np.array([replay[1] for replay in minibatch]), -1)
        rewards = np.array([replay[2] for replay in minibatch])
        next_s_batch = np.array([replay[3] for replay in minibatch])
        not_done = np.array([replay[4] for replay in minibatch])

        with tf.GradientTape() as tape:
            q_logits = self.model(s_batch)
            entropy_reg = tf.reduce_sum(-tf.nn.softmax(q_logits) * tf.nn.log_softmax(q_logits), -1)
            q_values = tf.gather_nd(q_logits, indices=selected_actions, batch_dims=1)
            target_q_values = tf.keras.backend.max(self.target_model(next_s_batch), -1)
            td_targets = tf.stop_gradient(rewards + 0.99 * target_q_values * not_done)
            td_loss = tf.compat.v1.losses.huber_loss(q_values, td_targets)
            total_loss = td_loss - self._entropy_coef * entropy_reg
            loss = tf.reduce_mean(input_tensor=total_loss)

        self.entropy(entropy_reg)
        self.td_loss(td_loss)
        self.train_loss(loss)

        # Gradient Clipping
        if self._max_norm != None:
            gradients = tape.gradient(loss, self.model.trainable_variables)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self._max_norm)
            self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_variables))
        else:
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Soft Update
        if self._soft_update:
            self._run_soft_update()

        # Peroidcally Update
        else:
            if self.step % self.update_freq == 0:
                self.target_model.set_weights(self.model.get_weights())

        # Save Model
        if self.step % self.save_freq == 0:
            self.save_model()

        # Write log to tensorboard
        if self.step % 100 == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar("train_loss", self.train_loss.result(), step=self.step)
                tf.summary.scalar("td_loss", self.td_loss.result(), step=self.step)
                tf.summary.scalar("entropy", self.entropy.result(), step=self.step)
                
def main(args):
    train(args)
    '''
    if not args.evaluate:
        train(args)
    else:
        eval(args)
    '''

def train(args):
    env = gym.make(args.environment)
    agent = DQN(state_dim=env.observation_space.shape, action_dim=env.action_space.n, lr=args.learning_rate)
    
    max_score = 0
    e_count = 0
    while True:
        # Training
        for _ in range(20):
            s = env.reset()
            score = 0
            while True:
                a = agent.act(s)
                next_s, reward, done, _ = env.step(a)
                agent.save_transition(s, a, reward, next_s, done)
                agent.train()
                score += reward
                s = next_s
                if done:
                    e_count += 1
                    max_score = max(max_score, score)
                    print('episode:', e_count, 'score:', score, 'max:', max_score)
                    break

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
        
        '''
        print ("Evaluated the agent for 5 episodes.")
        print ("Max: {}".format(max(score_list)))
        print ("Mean: {}".format(mean(score_list)))
        print ("Min: {}".format(min(score_list)))
        '''
        with agent.summary_writer.as_default():
            tf.summary.histogram("Evaluation Scores", score_list, step=agent.step)
        
    env.close()

'''
Note: Future Work
def eval(args):
    if args.checkpoint_path == None:
        print("Please provide the path of checkpoint you want to evaluate")
        return

    env = gym.make(args.environment)
    agent = DQN(state_dim=env.observation_space.shape, action_dim=env.action_space.n, lr=args.learning_rate)
    agent.restore(args.checkpoint_path)

    eval_episodes = 10
    score_list = []
    for _ in range(eval_episodes):
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

    print ("Evaluated the agent for {} episodes.".format(eval_episodes))
    print ("Max: {}".format(max(score_list)))
    print ("Mean: {}".format(mean(score_list)))
    print ("Min: {}".format(min(score_list)))
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', '--environment', help='The name of OpenAI Gym Envrionment. e.g. CartPole-v1', type=str, default="CartPole-v1")
    parser.add_argument('-lr', '--learning_rate', help='Learning Rate', type=float, default=0.0005)
    parser.add_argument('-su', '--soft_update', help='Whether to use soft-update', action="store_true")
    # parser.add_argument('-eval', '--evaluate', help="Evaluate mode", action="store_true")
    parser.add_argument('-cp', '--checkpoint_path', help="Checkpoint path", type=str, default=None)
    args = parser.parse_args()
    main(args)