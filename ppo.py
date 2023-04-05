import os
import numpy as np 
from model import Model, ICMModel
from arguments import Args
from datetime import datetime
from torch.optim import Adam
import pandas as pd
import torch
device = Args.device
import matplotlib.pyplot as plt
from normalizer import RewardScaling, State_Normalization

class PPO():
    def __init__(self, env, args) -> None:
        self.env = env
        self.args = args
        self.actor = Model(model_type = 'actor', args = args)
        self.critic = Model(model_type = 'critic', args = args)
        self.icm_model = ICMModel(self.args.Number_States, 1, self.args.ICM_hidden_dim)

        self.gamma  = args.gamma
        self.gae_lambda = args.gae_lambda
        self.reward_scaler = RewardScaling(shape = 1, gamma = args.gamma)
        self.State_Normalization = State_Normalization(shape=args.Number_States)
        self.optimizer = Adam([
                        {'params': self.actor.parameters(), 'lr': args.actor_lr},
                        {'params': self.critic.parameters(), 'lr': args.critic_lr}], eps=1e-5)
        self.icm_optimizer = Adam(self.icm_model.parameters(), lr=self.args.icm_lr)

        self.total_wait, self.total_queue = [], []
        self.output_dir='checkpoints'
        os.makedirs(self.output_dir, exist_ok=True)


        # self.memory = Memory() # PPO have no memory

    def train(self):
        test_reward = []
        test_neg_reward = []
        total_intrinsic_reward = []
        total_train_reward = []
        total_neg_reward = []
        for episode in range(self.args.total_episodes):
            print(f'----- Episode----------{episode * self.args.collects_cycles}---------{datetime.now()}-----------')
            data_container = {
                    'obs'  : [],
                    'actions' : [],
                    'old_log_prob' : [],
                    'rewards' : [],
                    'done' : [],
                    'info' : []
                }
            for n in range(self.args.collects_cycles):
                ep_intrinsic_reward = 0
                ep_obs, ep_actions, ep_log_prob, ep_reward, ep_done, ep_info = [], [], [], [], [], []
                obs = self.env.reset()
                done = False
                total_reward = 0
                neg_reward = 0
                while not done:
                    if self.args.state_normal:
                        obs = self.State_Normalization(obs)
                    action_dist = self.actor(torch.tensor(obs).float()).reshape(-1)
                    dist = torch.distributions.Categorical(logits = action_dist)
                    action = dist.sample().detach().cpu().item()
                    log_prob = action_dist[action]
                    new_obs, reward, done, info = self.env.step(action)
                    total_reward += reward
                    if reward<0:
                        neg_reward+=reward

                    reward = self.reward_scaler(reward) # scaling the reward
                    

                    # calculate instrinsic reward and add at the final reward
                    predicted_next_state, forward_loss, inverse_loss =\
                         self.icm_model(
                            torch.tensor(obs, dtype=torch.float32).unsqueeze(0), 
                            torch.tensor(action, dtype=torch.float32).unsqueeze(0).unsqueeze(0), 
                            torch.tensor(new_obs, dtype=torch.float32).unsqueeze(0)
                            )
                    intrinsic_reward = self.args.ICM_forward_loss_factor * forward_loss + self.args.ICM_inverse_loss_factor * inverse_loss
                    # print(intrinsic_reward.shape)
                    intrinsic_reward = intrinsic_reward.detach().cpu().numpy()  # convert tensor to scalar
                    # print(f'intrinsic_reward is {intrinsic_reward * self.args.intrinsic_reward_factor}, origin reward is {reward}')
                    reward += intrinsic_reward * self.args.intrinsic_reward_factor
                    ep_intrinsic_reward += intrinsic_reward * self.args.intrinsic_reward_factor
                    ep_obs.append(obs)
                    ep_actions.append(action)
                    ep_log_prob.append(log_prob)
                    ep_reward.append(reward)
                    ep_done.append(done)
                    ep_info.append(info)
                    if done:
                        
                        wait, queue = 0, 0
                        # for vehicle in self.env.vehicle.values():
                        #     wait += sum(vehicle.values())
                        # for car in self.env.vehicle_queue.values():
                        #     queue += sum(car.values())
                        wait += sum(self.env.total_waiting_time_episode)
                        queue +=sum(self.env._queue_length_episode)
                        self.total_wait.append(wait / 500)
                        self.total_queue.append(queue / 500)
                        total_train_reward.append(total_reward / 1000)
                        total_neg_reward.append(neg_reward / 1000)

                        data = {"等待时间：": self.total_wait, "队列长度": self.total_queue, '总负奖励':total_neg_reward}
                        pd.DataFrame(data).to_csv("./results/PPO_训练数据.csv", encoding='utf-8-sig')

                        ep_obs.append(obs)
                        ep_done.append(done)

                        plt.cla()
                        plt.plot(total_train_reward)
                        plt.savefig('./results/total_train_reward_ppo.png')

                        plt.cla()
                        plt.plot(total_neg_reward)
                        plt.savefig('./results/total_train_neg_reward_ppo.png')

                        plt.cla()
                        plt.plot(self.total_wait)
                        plt.savefig('./results/total_waiting_time_ppo.png')

                        plt.cla()
                        plt.plot(self.total_queue)
                        plt.savefig('./results/total_queue_length_ppo.png')
                        break
                    obs = new_obs
                total_intrinsic_reward.append(ep_intrinsic_reward)
                data_container['obs'].append(ep_obs)
                data_container['actions'].append(ep_actions)
                data_container['rewards'].append(ep_reward)
                data_container['done'].append(ep_done)
                data_container['info'].append(ep_info)
                data_container['old_log_prob'].append(ep_log_prob)
            training_data =  self.prepare_update_data(data_container)             
            self.update_network(training_data)
            self.update_icm(training_data)
            self.lr_decay(episode)
            if episode and episode % self.args.test_interval == 0:
                average_reward, average_neg_reward = self.test_agent()
                test_reward.append(average_reward)
                test_neg_reward.append(average_neg_reward)
                torch.save(self.actor.state_dict(), os.path.join(self.output_dir, f'policy_model_{episode}.pt'))
                print(f"average test reward: {average_reward}, average neg reward {average_neg_reward}") #.format()格式化打印的表达。Z
            plt.plot(test_reward)
            plt.savefig('reward_Curl.png')
            # plt.cla()
            # plt.plot(test_neg_reward)
            # plt.savefig('neg_reward_Curl.png')
            # plt.cla()
            # plt.plot(total_intrinsic_reward)
            # plt.savefig('total_intrinsic_reward.png')
            # plt.cla()


    def lr_decay(self, total_steps):
        lr_a_now = self.args.actor_lr * (1 - total_steps / self.args.total_episodes)
        lr_c_now = self.args.critic_lr * (1 - total_steps / self.args.total_episodes)
        self.optimizer.param_groups[0]['lr']  = lr_a_now
        self.optimizer.param_groups[1]['lr']  = lr_c_now

    def test_agent(self):
        total_rewards = []
        total_neg_reward = []
        for episode in range(self.args.test_episode):
            tot_neg_reward = 0 #初始化总的延迟奖励
            tot_reward = 0
            obs = self.env.reset()
            done = False
            while not done:
                action = self.choose_action((torch.tensor(obs).float()))
                new_obs, reward, done, info = self.env.step(action)
                tot_reward += reward
                if reward < 0:
                    tot_neg_reward += reward
                if done: 
                    total_rewards.append(tot_reward)
                    total_neg_reward.append(tot_neg_reward)
                    break
                obs = new_obs
        return sum(total_rewards) / len(total_rewards), sum(total_neg_reward) / len(total_neg_reward)

    def choose_action(self, state):
        action = torch.argmax(self.actor(torch.tensor(state).float()).reshape(-1)).detach().cpu().item()
        return action #Use memory，维持当前


    def prepare_update_data(self, data):
        values, returns,  td_targets, advs, next_states = [], [], [], [], []
        collected_eps = len(data['obs'])
        for i in range(collected_eps):
            value = self.critic(torch.tensor(np.array(data['obs'][i]), dtype = torch.float32)).reshape(-1).detach().cpu().numpy()
            data['obs'][i] = np.array(data['obs'][i][:-1])
            next_states.append(np.array(data['obs'][i][1:]))
            data['actions'][i] = np.array(data['actions'][i])
            data['rewards'][i] = np.array(data['rewards'][i])
            data['done'][i] = np.array(data['done'][i])
            # breakpoint()
            t = [item.detach().numpy() for item in data['old_log_prob'][i]]
            data['old_log_prob'][i] = np.array(t)
        # rewards = (rewards) / self.args.reward_scaling
            advantages, td_target, ep_return = self.compute_GAE(data['rewards'][i], value, data['done'][i])
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            values.append(value)
            td_targets.append(td_target)
            advs.append(advantages)
            returns.append(ep_return)
        return {
            'obs': np.concatenate(data['obs'], axis = 0),
            'actions': np.concatenate(data['actions'], axis = 0),
            'rewards': np.concatenate(data['rewards'], axis = 0).reshape(-1),
            'old_log_prob': np.concatenate(data['old_log_prob'], axis = 0),
            'advantages': np.concatenate(advs, axis = 0).reshape(-1),
            'td_target': np.concatenate(td_targets, axis = 0).reshape(-1),
            'returns': np.concatenate(returns, axis = 0).reshape(-1),
            'next_values': np.concatenate(next_states, axis = 0)
        }

    def update_network(self, data):
        # Train for K epochs
        total_loss = []
        for _ in range(self.args.training_time_per_episode):
        # Perform updates on minibatches
            loss = 0
            for batch in self._split_data(data, self.args.batch_size):
                obs_batch, act_batch, old_log_probs_batch, adv_batch, td_target, returns =\
                batch['obs'], batch['actions'], batch['old_log_probs'], batch['advantages'], batch['td_target'],batch['returns']
                # Compute loss and update parameters
                act_batch = torch.tensor(act_batch, dtype = torch.float32)
                adv_batch = torch.tensor(adv_batch, dtype = torch.float32)
                td_target= torch.tensor(td_target, dtype = torch.float32)
                
                returns= torch.tensor(returns, dtype = torch.float32)
                old_log_probs_batch= torch.tensor(old_log_probs_batch.astype(np.float32), dtype = torch.float32)
                new_action_logits = self.actor(torch.tensor(obs_batch, dtype = torch.float32))
                new_action_dist = torch.distributions.Categorical(logits=new_action_logits)
                new_log_prob_batch = new_action_dist.log_prob(act_batch.long())
                entropy_item = new_action_dist.entropy().mean()
                
                # new_log_prob_batch = torch.gather(new_action_logits, dim = 1, index= act_batch.long().reshape(-1,1))
                
                ratio = torch.exp(new_log_prob_batch - old_log_probs_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * adv_batch
                critic_loss = (self.critic(torch.tensor(obs_batch, dtype = torch.float32))  - returns ).pow(2)
                # critic_loss = torch.nn.functional.mse_loss(self.critic(torch.tensor(obs_batch, dtype = torch.float32)), returns)

                # actor_loss =  adv_batch * (new_log_prob_batch)
                # loss = actor_loss.mean() + self.args.critic_coef * critic_loss.mean() - self.args.entropy_coef * entropy_item
            
                loss = -torch.min(surr1, surr2).mean() + self.args.critic_coef * critic_loss.mean() - self.args.entropy_coef * entropy_item
                self.optimizer.zero_grad()
                loss.backward()
                if self.args.use_grad_clip: # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) 
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) 
                self.optimizer.step()
                total_loss.append(loss.mean().item())
        # print(f'learning loss : {sum(total_loss)/len(total_loss)}')
                
    # def reward_normalize(self, reward_data):
    #     return (reward_data - self.reward_mean) / self.reward_std
     
    # def update_normalizer(self, reward):

    def update_icm(self, data):
        total_icm_loss = 0
        for batch in self._split_data(data, self.args.batch_size):
            obs_batch = batch['obs']
            action_batch = batch['actions']
            next_states_batch = batch['next_values']
            # compute ICM loss
            predicted_next_state, forward_losses, inverse_losses =\
                         self.icm_model(
                            torch.tensor(obs_batch, dtype=torch.float32), 
                            torch.tensor(action_batch, dtype=torch.float32).unsqueeze(1), 
                            torch.tensor(next_states_batch, dtype=torch.float32)
                            )
            intrinsic_rewards = self.args.ICM_forward_loss_factor * forward_losses + self.args.ICM_inverse_loss_factor * inverse_losses # 
            intrinsic_rewards = intrinsic_rewards.mean()
            icm_loss = intrinsic_rewards
            
            # optimize ICM model
            self.icm_optimizer.zero_grad()
            icm_loss.backward()
            self.icm_optimizer.step()
            total_icm_loss += icm_loss.item()
        print(f'total icm loss {total_icm_loss}')

    def compute_GAE(self, rewards, values, done):
        mb_advs = np.zeros_like(rewards)
        td_target = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):  # 倒序实现，便于进行递归
            nolast = 1.0 - done[t + 1]
            if t == len(rewards) - 1:  # 如果是最后一步，要判断当前是否是终止状态，如果是，next_value就是0
                returns[t]  = rewards[t]
            else:
                returns[t]  = rewards[t] + self.gamma * returns[t+1]
            nextvalue = values[t+1]
            delta = rewards[t] + nolast * self.gamma * nextvalue - values[t]
            mb_advs[t] = lastgaelam = delta + nolast * self.gamma * self.gae_lambda * lastgaelam
        return mb_advs, td_target, returns
    
    def _split_data(self, data, batch_size):
        # Split data into minibatches of size batch_size
        #将数据拆分成大小为batch_size的小批量
        num_batches = len(data['obs']) // batch_size
        batches = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch = {
                'obs': data['obs'][start_idx:end_idx],
                'actions': data['actions'][start_idx:end_idx],
                'old_log_probs': data['old_log_prob'][start_idx:end_idx],
                'advantages': data['advantages'][start_idx:end_idx],
                'td_target' : data['td_target'][start_idx:end_idx],
                'returns' : data['returns'][start_idx:end_idx],
                'next_values' : data['next_values'][start_idx:end_idx]
            }
            batches.append(batch)
        return batches
