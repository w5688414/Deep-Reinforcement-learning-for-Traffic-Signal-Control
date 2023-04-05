import numpy as np 
import math
import time
import random
#Class for traffic generation
#定义生成交通的累

class TrafficGenerator:

    def __init__(self, Max_Steps, Total_Number_Cars = 500): #在最大时间步参数下的交通生成,Max_Steps=3600s
        self.Total_Number_Cars = Total_Number_Cars  #Number of cars used in the simulation 在模拟中使用的汽车数量 500是否小了？（可以修改数值为高流量密度和低流量密度）
        self._max_steps = Max_Steps #训练最大步数

    def generate_routefile(self):
        random.seed(int(time.time()))
        np.random.seed(int(time.time())) #设置随机种子，当下次设置随机种子之后，产生相同的随机数。
        Timing = np.sort(np.random.poisson(2, self.Total_Number_Cars)) #车辆服从泊松分布（可以设置为其他分布，而且，可以想办法刻画车辆生成的统计图模型） Poisson distribution for the car approach rate to teh intersection
        Car_Generation_Steps = []
        min_old = math.floor(Timing[1]) #返回小于的最小整数值，车的最小值
        max_old = math.ceil(Timing[-1]) #返回大于的最大整数值，车的最大值
        min_new = 0
        max_new = self._max_steps
        #Create .xml file for SUMO simulation　
        for value in Timing:
            Car_Generation_Steps = np.append(Car_Generation_Steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new) #拼接生成车辆生成时间步
        Car_Generation_Steps = np.rint(Car_Generation_Steps)  #四舍五入取整

        with open("project.rou.xml", "w") as routes: #打开路由文件
            #Generate Routes
            #生成的路线
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            <route id="W_N" edges="1i 4o"/>
            <route id="W_E" edges="1i 2o"/>
            <route id="W_S" edges="1i 3o"/>
            <route id="N_W" edges="4i 1o"/>
            <route id="N_E" edges="4i 2o"/>
            <route id="N_S" edges="4i 3o"/>
            <route id="E_W" edges="2i 1o"/>
            <route id="E_N" edges="2i 4o"/>
            <route id="E_S" edges="2i 3o"/>
            <route id="S_W" edges="3i 1o"/>
            <route id="S_N" edges="3i 4o"/>
            <route id="S_E" edges="3i 2o"/>""", file=routes)

            #Generate cars to follow routes
            #生成汽车遵循路线
            for car_counter, step in enumerate(Car_Generation_Steps): #将车辆生成的时间步按索引遍历，名字car_counter，索引值step
                Straight_or_Turn = np.random.uniform() #给直行或者转弯随即初始值
                Straight = np.random.randint(1,9) #随机生成1~9之间的整数（用于直行和右转）
                route = {
                    1: "W_E",
                    2: "E_W",
                    3: "N_S",
                    4: "S_N",
                    5: "W_S",
                    6: "W_N",
                    7: "E_S",
                    8: "E_N"
                }
                print(f'    <vehicle id="{route[Straight]}_{car_counter}" type="standard_car" route="{route[Straight]}" depart="{step}" departLane="random" departSpeed="10" />', file=routes)
            else:
                routes_names = ["S_E", "S_W", "N_W", "N_E"]
                turn = np.random.randint(0, 4)
                route_name = routes_names[turn]
                vehicle_id = f"{route_name}_{car_counter}"
                print(f'    <vehicle id="{vehicle_id}" type="standard_car" route="{route_name}" depart="{step}" departLane="random" departSpeed="10" />', file=routes)
            print("</routes>", file=routes)
