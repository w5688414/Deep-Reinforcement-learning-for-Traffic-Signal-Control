import pandas as pd
import os, sys
from arguments import Args
from env import traffic_env


tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

def Fix(env):
    Actions = [1, 4, 7, 10] # 此处对应的是你env中的动作，在下面循环使用
    total_wait, total_queue = [], []

    for l in range(1000):
        obs = env.reset()
        i = 0
        while True:
            act = Actions[i]
            next_obs, reward, done, info = env.step(act)
            if i == 3:
                i = 0
            i += 1
            if done:
                wait, queue = 0, 0
                # for vehicle in env.vehicle.values():
                #     wait += sum(vehicle.values())
                # for car in env.vehicle_queue.values():
                #     queue += sum(car.values())

                wait += sum(env.total_waiting_time_episode)
                queue +=sum(env._queue_length_episode)
                total_wait.append(wait / 500)
                total_queue.append(queue / 500)
                data = {"等待时间：": total_wait, "队列长度": total_queue}
                pd.DataFrame(data).to_csv("./results/Fix_数据.csv", encoding='utf-8-sig')

                break


def main():
    args = Args()
    env = traffic_env(args.Green_Duration, args.Yellow_Duration, args.SUMO_Command)
    Fix(env)

if __name__ == '__main__':
    main()