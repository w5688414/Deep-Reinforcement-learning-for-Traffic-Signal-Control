#Sanil Lala

from __future__ import absolute_import
from __future__ import print_function
import os
import argparse
import sys
from ppo import PPO
from advantage_actor_critic import advantage_actor_critic
from arguments import Args
from env import traffic_env
from dqn import Deep_Q_network

#Set SUMO environment path and import SUMO library and Traci
#设置SUMO环境路径，导入SUMO库和Traci
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='dqn', help="Algorithms, eg PPO, DQN, A2C")
input_args = parser.parse_args()

def main():
    args = Args() 
    os.makedirs('results', exist_ok=True)
    env = traffic_env(args.Green_Duration, args.Yellow_Duration, args.SUMO_Command)
    
    if input_args.model_name == 'ppo':
        PPO_agent = PPO(env, args=args)
        PPO_agent.train()
    elif input_args.model_name =='a2c':
        A2C_agent = advantage_actor_critic(env, gamma=0.99, learning_rate=1e-3, episode=1000, args=args)
        A2C_agent.run()
    elif input_args.model_name =='dqn':
        dqn_agent = Deep_Q_network(env, args.gamma, args.learning_rate, args.total_episodes, args.tau, args=args)
        dqn_agent.run()
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    main()

