import traci
import numpy as np
from generator_env import TrafficGenerator
from arguments import Args

max_step = Args.Max_Steps


class traffic_env:
    def __init__(self, Green_Duration, Yellow_Duration, SUMO_Command):
        self.traffic_gen = TrafficGenerator(max_step)
        self._SUMO_Command = SUMO_Command
        self._max_steps = max_step
        self._green_duration = Green_Duration
        self._yellow_duration = Yellow_Duration
        self._sum_intersection_queue = 0  # 总的交叉口排队长度
        self.last_total_wait_time = 0

        self.vehicle = dict()
        self.vehicle_queue = dict()
        self._queue_length_episode = []
        self.total_waiting_time_episode=[]

    def reset(self):
        self.last_action = -1
        self.traffic_gen.generate_routefile()  # 获得车辆生成
        traci.start(self._SUMO_Command)
        self.lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes("0")))
        self.steps = 0
        self.last_total_wait_time = 0
        obs = self._get_state()

        self.vehicle = dict()
        self.vehicle_que = dict()

        self._queue_length_episode = []
        self.total_waiting_time_episode=[]

        return obs

    def step(self, action):
        # 当前交通信号与先前不同时，设置黄色相位
        if self.last_action != action:  # 仿真步不是0并且动作和上一次的动作不一致
            self._Set_YellowPhase(self.last_action)  # 根据当前的动作设置一个黄色相位
            self._simulate(self._yellow_duration)
        self._Set_GreenPhaseandDuration(action)
        reward = self._simulate(self._green_duration)
        obs = self._get_state()  # 获取状态作为当前的状态（共有包含四部分的状态选取，位置，速度，相位，相位时间）
        # print(reward)
        self.last_action = action
        done = False if self.steps != self._max_steps else True  # maybe always false?
        if done:
            traci.close()
        info = {
            'cur_step': self.steps
        }
        return obs, reward, done, info

    def _simulate(self, duration_time):
        duration_time = min(self._max_steps - self.steps, duration_time)
        sum_waiting_time = 0
        self.steps += duration_time
        for _ in range(duration_time):
            traci.simulationStep()  # 创建模拟步骤
            # intersection_queue = self._get_waiting_times()
            # sum_waiting_time += intersection_queue
            waiting_time = self.get_accumulated_waiting_time_per_lane()
            sum_waiting_time+=waiting_time
            self.total_waiting_time_episode.append(waiting_time)

            queue_length = self._get_queue_length() 
            self._queue_length_episode.append(queue_length)


            # self.get_queue_per_lane()
        # print(f'cur step simulation {duration_time}, sum waiting time {sum_waiting_time/100}')
        # return - (sum_waiting_time / 100)
        # Get rewards
        reward = self.last_total_wait_time - sum_waiting_time
        # Store current waiting time
        self.last_total_wait_time = sum_waiting_time

        return reward

    def _get_queue_length(self):
        halt_N = traci.edge.getLastStepHaltingNumber("1i")
        halt_S = traci.edge.getLastStepHaltingNumber("2i")
        halt_E = traci.edge.getLastStepHaltingNumber("3i")
        halt_W = traci.edge.getLastStepHaltingNumber("4i")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    # def _get_waiting_times(self):
    #     intersection_queue = 0
    #     loop_time = 0
    #     # 1，2，3，4分别表示东西南北
    #     for loop in [1, 2, 3, 4]:
    #         # 有两条道，分别用下面的0和1表示
    #         for j in [0, 1]:
    #             # 1，2，3表示的是线圈的代表号
    #             for k in [1, 2, 3]:
    #                 intersection_queue += traci.inductionloop.getLastStepOccupancy(f"Loop{loop}i_{j}_{k}")
    #                 loop_time += 1
    #     # print( intersection_queue/ loop_time)
    #     # return average occupy
    #     return intersection_queue / loop_time

    # set yellow phase
    def _Set_YellowPhase(self, old_action):  # 根据旧的动作设置黄色相位。
        old_action_to_yellow = {
            "0, 1, 2": 1,
            "3, 4, 5": 3,
            "6, 7, 8": 5,
            "9, 10, 11": 7,
        }
        yellow_phase = 1
        for old_actions, value in old_action_to_yellow.items():
            if old_action in list(map(int, old_actions.strip().split(','))):  # convert string to list
                yellow_phase = value
                break
        traci.trafficlight.setPhase("0", yellow_phase)

    # set green phase duration
    def _Set_GreenPhaseandDuration(self, action):
        action_to_traffic_setting = {
            0: [0, 15], 1: [0, 10], 2: [0, 20], 3: [2, 15],
            4: [2, 10], 5: [2, 20], 6: [4, 15], 7: [4, 10],
            8: [4, 20], 9: [6, 15], 10: [6, 10], 11: [6, 20],
        }
        params = action_to_traffic_setting[action]  # 相位、相位持续时间、标记
        traci.trafficlight.setPhase("0", params[0])  # 相位为0
        traci.trafficlight.setPhaseDuration("0", params[1])  # 相位持续的时间为15
        self._green_duration = params[1]

    # obtain state after action 动作执行后获取下一时刻状态
    def _get_state(self):
        # Create 12 x 3 Matrix for Vehicle Positions and Velocities  #这里往往可以加入当前红绿灯状态或者动作选择  #创建12 × 3矩阵的车辆位置和速度
        Position_Matrix = np.zeros([8, 3])  # 一个是直行的循环遍历 #一个是转弯的循环遍历
        Velocity_Matrix = np.zeros([8, 3])
        # get loop name
        loops = [
            "Loop1i_0_1", "Loop1i_0_2", "Loop1i_0_3", "Loop1i_1_1", "Loop1i_1_2", "Loop1i_1_3",
            "Loop2i_0_1", "Loop2i_0_2", "Loop2i_0_3", "Loop2i_1_1", "Loop2i_1_2", "Loop2i_1_3",
            "Loop3i_0_1", "Loop3i_0_2", "Loop3i_0_3", "Loop3i_1_1", "Loop3i_1_2", "Loop3i_1_3",
            "Loop4i_0_1", "Loop4i_0_2", "Loop4i_0_3", "Loop4i_1_1", "Loop4i_1_2", "Loop4i_1_3"]
        for i, loop in enumerate(loops):
            loop_vehicle_ids = traci.inductionloop.getLastStepVehicleIDs(loop)
            if len(loop_vehicle_ids) != 0:
                Velocity_Matrix[i // 3, i % 3] = traci.vehicle.getSpeed(loop_vehicle_ids[0])
                Position_Matrix[i // 3, i % 3] = 1
            else:
                Position_Matrix[i // 3, i % 3] = 0
        # Create 4 x 1 matrix for phase state
        # 创建4 x 1矩阵的相位状态
        Phase = []
        # 定义交通灯相位对应的列表
        phase_list = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        # 获取当前交通灯相位并根据相位值选择相位列表中的对应项
        traffic_light_phase = traci.trafficlight.getPhase('0')
        if traffic_light_phase in range(12):
            Phase = phase_list[traffic_light_phase // 3]

        state = np.concatenate((Position_Matrix, Velocity_Matrix), axis=0).flatten()  # 完成位置矩阵，速度矩阵拼接。
        state = np.concatenate((state, np.array(Phase).flatten()), axis=0)  # 将速度位置状态和相位拼接作为最终要送给神经网络的状态
        # 创建持续时间矩阵
        Duration_Matrix = np.array([traci.trafficlight.getPhaseDuration('0')]).flatten()  # '0'是干什么的？获取已经持续的相位时间
        state = np.concatenate((state, Duration_Matrix), axis=0)  # 上边的state加上持续时间，共有四个输入（位置，速度，相位，相位时间）作为state。
        return state

    def get_accumulated_waiting_time_per_lane(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ['4i', '2i','3i', '1i']
        # incoming_roads = self.lanes
        car_list = traci.vehicle.getIDList()
        # print(car_list)
        # print(incoming_roads)
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            # print(wait_time)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            # print(road_id)
            # print(type(road_id))
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                # print(road_id)
                self.vehicle[car_id] = wait_time
            else:
                if car_id in self.vehicle: # a car that was tracked has cleared the intersection
                    del self.vehicle[car_id] 
        total_waiting_time = sum(self.vehicle.values())
        # print(total_waiting_time)
        return total_waiting_time

    # def get_accumulated_waiting_time_per_lane(self):
    #     wait_time_per_lane = []

    #     for lane in self.lanes:
    #         veh_list = traci.lane.getLastStepVehicleIDs(lane)
    #         wait_time = 0.0

    #         for veh in veh_list:
    #             veh_lane = traci.vehicle.getLaneID(veh)
    #             acc = traci.vehicle.getAccumulatedWaitingTime(veh)
    #             if veh not in self.vehicle:
    #                 self.vehicle[veh] = {veh_lane: acc}
    #             else:
    #                 self.vehicle[veh][veh_lane] = acc - sum(
    #                     [self.vehicle[veh][lane] for lane in self.vehicle[veh].keys() if lane != veh_lane])
    #             wait_time += self.vehicle[veh][veh_lane]
    #         wait_time_per_lane.append(wait_time)
    #     total_waiting_time = sum(wait_time_per_lane)
    #     return total_waiting_time

    # def get_queue_per_lane(self):
    #     for lane in self.lanes:
    #         veh_list = traci.lane.getLastStepVehicleIDs(lane)
    #         for veh in veh_list:
    #             veh_lane = traci.vehicle.getLaneID(veh)
    #             v = traci.vehicle.getSpeed(veh)
    #             if v < 0.1:
    #                 if veh not in self.vehicle_queue:
    #                     self.vehicle_queue[veh] = {veh_lane: 1}
    #                 else:
    #                     if veh_lane not in self.vehicle_queue[veh]:
    #                         self.vehicle_queue[veh] = {veh_lane: 1}
    #                     else:
    #                         self.vehicle_queue[veh][veh_lane] += 1

