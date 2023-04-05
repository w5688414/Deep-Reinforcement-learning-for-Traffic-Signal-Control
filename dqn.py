

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from arguments import Args

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Deep_Q_network(object):

    def __init__(self, env, gamma, learning_rate, episode, tau, args, render=False):
        self.env = env
        self.args = args
        self.observation_dim = self.args.Number_States
        self.action_dim = self.args.Number_Actions
        self.learning_rate = learning_rate
        self.batch_size = self.args.batch_size 
        self.gamma = gamma
        self.tau = tau
        self.episode = episode
        self.render = render

        self.steps_done = 0

        self.policy_net = Net(self.observation_dim, self.action_dim)
        self.target_net = Net(self.observation_dim, self.action_dim)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)

        self.memory = ReplayMemory(10000)
        self.total_wait, self.total_queue = [], []
        self.episode_rewards=[]
        self.episode_neg_rewards = []

    def train(self, ):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]


        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()



    def run(self, ):
        for i in range(self.episode):
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            total_reward = 0
            total_neg_reward = 0
            if self.render:
                self.env.render()
            
            for t in count():

                sample = random.random()
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * self.steps_done / EPS_DECAY)
                self.steps_done += 1
                if sample > eps_threshold:
                    with torch.no_grad():
                        # t.max(1) will return the largest column value of each row.
                        # second column on max result is index of where max element was
                        # found, so we pick action with the larger expected reward.
                        action = self.policy_net(state).max(1)[1].view(1, 1)
                else:
                    idx = random.randint(0, self.args.Number_Actions-3)
                    action = torch.tensor([[idx]], dtype=torch.long)

                observation, reward, terminated, info = self.env.step(action.item())
                if reward < 0:
                    # tot_neg_reward += reward #总的奖励值
                    total_neg_reward += reward
                total_reward += reward
                reward = torch.tensor([reward])

                done = terminated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)
                # Move to the next state
                state = next_state
                # Perform one step of the optimization (on the policy network)
                self.train()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    # episode_rewards.append(total_reward)
                    wait, queue = 0, 0
                    # for vehicle in self.env.vehicle.values():
                    #     wait += sum(vehicle.values())
                    # for car in self.env.vehicle_queue.values():
                    #     queue += sum(car.values())
                    wait += sum(self.env.total_waiting_time_episode)
                    queue +=sum(self.env._queue_length_episode)

                    self.episode_rewards.append(total_reward / 1000)
                    self.total_wait.append(wait / 500)
                    self.total_queue.append(queue / 500)
                    self.episode_neg_rewards.append(total_neg_reward / 1000)

                    print('episode: {}  reward: {:.1f}  total_neg_reward: {:.2f}'.format(i+1, total_reward, total_neg_reward))

                    data = {"等待时间：": self.total_wait, "队列长度": self.total_queue, '总负奖励':self.episode_neg_rewards}
                    pd.DataFrame(data).to_csv("./results/DQN_训练数据.csv", encoding='utf-8-sig')
                    # plot_durations()
                    model_type='dqn'
                    plt.cla()
                    plt.plot(self.episode_rewards)
                    plt.savefig('./results/total_train_reward_{}.png'.format(model_type))

                    plt.cla()
                    plt.plot(self.episode_neg_rewards)
                    plt.savefig('./results/total_train_neg_reward_{}.png'.format(model_type))

                    plt.cla()
                    plt.plot(self.total_wait)
                    plt.savefig('./results/total_waiting_time_{}.png'.format(model_type))

                    plt.cla()
                    plt.plot(self.total_queue)
                    plt.savefig('./results/total_queue_length_{}.png'.format(model_type))
                    break


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    episode = 50
    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)
    args = Args() 

    args.Number_States = n_observations
    args.Number_Actions = n_actions

    dqn_agent = Deep_Q_network(env, GAMMA, LR, episode, TAU, args)
    dqn_agent.run()

    print('Complete')
    # plot_durations(show_result=True)
    plt.ioff()
    plt.show()
    plt.savefig('dqn_out.png')