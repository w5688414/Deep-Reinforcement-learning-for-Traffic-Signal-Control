import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt

class policy_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(policy_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

        self.rewards = []
        self.log_probs = []

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return F.softmax(x, 1)

    def act(self, input):
        probs = self.forward(input)
        dist = Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.log_probs.append(log_prob)
        return action[0].item()


class value_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(value_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class advantage_actor_critic(object):
    def __init__(self, env, gamma, learning_rate, episode, args, render=False):
        self.env = env
        self.args = args
        self.observation_dim = self.args.Number_States
        self.action_dim = self.args.Number_Actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.episode = episode
        self.render = render
        self.policy_net = policy_net(self.observation_dim, self.action_dim)
        self.value_net = value_net(self.observation_dim, 1)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.total_returns = []
        self.values_buffer = []
        self.writer = SummaryWriter('runs/a2c')
        self.weight_reward = None
        self.count = 0

        self.total_wait, self.total_queue = [], []

        self.output_dir='checkpoints'
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, ):
        total_returns = torch.FloatTensor(self.total_returns).unsqueeze(1).detach()
        values = torch.cat(self.values_buffer, 0)
        delta = (total_returns - values).squeeze(1)
        log_probs = torch.cat(self.policy_net.log_probs, 0)

        policy_loss = (- log_probs * delta.detach())
        policy_loss = policy_loss.sum()
        self.writer.add_scalar('policy_loss', policy_loss, self.count)
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.1)
        self.policy_optimizer.step()

        value_loss = delta.pow(2).sum()
        self.writer.add_scalar('value_loss', value_loss, self.count)
        self.value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.1)
        self.value_optimizer.step()

    def run(self, ):
        total_train_reward = []
        total_neg_reward = []
        for i in range(self.episode):
            obs = self.env.reset()
            total_reward = 0
            neg_reward = 0
            if self.render:
                self.env.render()
            while True:
                self.values_buffer.append(self.value_net.forward(torch.FloatTensor(np.expand_dims(obs, 0))))
                action = self.policy_net.act(torch.FloatTensor(np.expand_dims(obs, 0)))
                next_obs, reward, done, info = self.env.step(action)
                self.policy_net.rewards.append(reward)
                self.count += 1
                total_reward += reward
                if reward<0:
                    neg_reward+=reward
                if self.render:
                    self.env.render()
                obs = next_obs
                if done:
                    R = 0
                    if self.weight_reward:
                        self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward
                    else:
                        self.weight_reward = total_reward
                    for r in reversed(self.policy_net.rewards):
                        R = R * self.gamma + r
                        self.total_returns.append(R)

                    wait, queue = 0, 0
                    # for vehicle in self.env.vehicle.values():
                    #     wait += sum(vehicle.values())
                    # for car in self.env.vehicle_queue.values():
                    #     queue += sum(car.values())

                    wait += sum(self.env.total_waiting_time_episode)
                    queue +=sum(self.env._queue_length_episode)
                    
                    self.total_wait.append(wait/500)
                    self.total_queue.append(queue/500)
                    total_train_reward.append(total_reward / 1000)
                    total_neg_reward.append(neg_reward / 1000)

                    data = {"等待时间：": self.total_wait, "队列长度": self.total_queue, '总负奖励':total_neg_reward}
                    pd.DataFrame(data).to_csv("./results/A2C_训练数据.csv", encoding='utf-8-sig')

                    self.total_returns = list(reversed(self.total_returns))
                    self.train()
                    del self.policy_net.rewards[:]
                    del self.policy_net.log_probs[:]
                    del self.total_returns[:]
                    del self.values_buffer[:]
                    print('episode: {}  reward: {:.1f}  weight_reward: {:.2f}'.format(i+1, total_reward, self.weight_reward))

                    plt.cla()
                    plt.plot(total_train_reward)
                    plt.savefig('./results/total_train_reward_a2c.png')

                    plt.cla()
                    plt.plot(total_neg_reward)
                    plt.savefig('./results/total_train_neg_reward_a2c.png')

                    plt.cla()
                    plt.plot(self.total_wait)
                    plt.savefig('./results/total_waiting_time_a2c.png')

                    plt.cla()
                    plt.plot(self.total_queue)
                    plt.savefig('./results/total_queue_length_a2c.png')
                    break


if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    test = advantage_actor_critic(env, gamma=0.99, learning_rate=1e-3, episode=100000, render=True)
    test.run()