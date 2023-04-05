# AI-Traffic-Control-System
This project was completed in partial fulfilment of my engineering degree.

## **Outline:**

The system utilises SUMO traffic simulator and Python 3 with TensorFlow. The system is developed for a minor arterial road intersection 
with traffic lights with turning signals. Utilising the reinforcement learning algorithm called Deep Q-learning, the system tries to choose the optimal traffic duration and phase to minimise traffic waiting 
times around the intersection. 
A 7 second yellow interval was employed, and green durations were adjusted between 10 s, 15 s and 20 s, depending on the vehicle demand. 
This system is a modified and adapted system developed by [1] as well as extracts from work done by [2, 3]. 
A more realistic approach was undertaken when developing this system with regards to real world sensors and data. 
Induction loop sensors were considered and thus the data from these sensors is what is used to drive the system. 

## environment setup

```
export SUMO_HOME=/usr/share/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

```

## **The Agent:**

**Framework:** Deep Q-Learning

```

python main.py --model_name dqn
```

**Framework:** PPO

```

python main.py --model_name ppo
```

**Framework:** A2C

```

python main.py --model_name a2c
```




## **Requirements to run the code:**
•	Python 

•	Pytorch 

•	SUMO Traffic simulator

•	Traffic Control Interface (TraCI) – this is included with SUMO

## **Additional files for the traffic generation and intersection layout:**
•	Add.xml – This file for the induction loops and initialling the traffic light phases.

•	Rou.xml – This file is created when running the code. It is for the vehicle routes and the paths in the simulation.

•	Con.xml – This file is for the round connections in the simulations.

•	Edg.xml – This is for the lanes.

•	Nod.xml – This is for the state and end points for the roads.

•	Net.xml – This is a configuration file to combine all the above files and create the road network.

•	Netccfg – This is a sumo network configuration.

•	Sumocfg – This is GUI file for the simulation


## **References:** 
1.	Vidali A, Crociani L, Vizzari G, Bandini,S, (2019). Bandini. A Deep Reinforcement Learning Approach to Adaptive Traffic Lights Management [cited 23 August 2019]. Available from: http://ceur-ws.org/Vol-2404/paper07.pdf
2.	Gao J, Shen Y, Liu J, Ito M and Shiratori N. Adaptive Traffic Signal Control: Deep Reinforcement Learning Algorithm with Experience Replay and Target Network. [Internet]. Arxiv.org. 2019 [cited 28 June 2019]. Available from: https://arxiv.org/pdf/1705.02755.pdf
3.	Liang X, Du X, Wang G, Han Z. (2018). Deep Reinforcement Learning for Traffic Light Control in Vehicular Networks. [cited 10 July 2019]. Available from: https://www.researchgate.net/publication/324104685_Deep_Reinforcement_Learning_for_Traffic_Light_Control_in_Vehicular_Networks
4.	DLR - Institute of Transportation Systems - Eclipse SUMO – Simulation of Urban MObility [Internet]. Dlr.de. 2019 [cited 10 July 2019]. Available from: https://www.dlr.de/ts/en/desktopdefault.aspx/tabid-9883/16931_read-41000/





