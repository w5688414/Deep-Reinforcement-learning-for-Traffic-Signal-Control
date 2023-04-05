from sumolib import checkBinary

class Args:
    #########################
    # training setting 
    #########################
    gui =False #如果是True，则不启动可视化界面
    total_episodes = 5000
    collects_cycles = 5
    test_interval = 1
    test_episode = 1
    batch_size = 512
    # device = 'cuda' if  torch.cuda.is_available() else 'cpu'
    device =  'cpu'
    #########################
    # PPO params
    #########################
    training_time_per_episode = 3 # PPO reuse
    clip_ratio = 0.2  # PPO Clip
    entropy_coef = 0.01 # PPO entropy
    state_normal = False  # use state normalizition
    use_grad_clip = True  # PPO use gradient clip 
    use_orthogonal_init = True # PPO use network orthogonal initial
    gae_lambda = 0.98 # GAE lambda
    critic_coef = 0.5 # PPO Critic loss factor
    #########################
    # common hyperparams
    #########################
    gamma = 0.98 #可以被修改的值
    hidden_layer = 2 
    hidden_size = 256
    actor_lr = 1e-3
    critic_lr = 1e-3
    Number_Actions = 12
    learning_rate = 1e-4
    tau = 0.005
    #########################
    # env setting
    #########################
    output_type = 'descrete' 
    Number_States = 53 #车辆状态
    Max_Steps = 3600 #仿真最大时间步
    Green_Duration = 10
    Yellow_Duration = 7 #黄色持续时间  4s
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui') #启动图形化仿真
    SUMO_Command = [sumoBinary, "-c", "project.sumocfg", '--start', "--no-step-log", "true", "--waiting-time-memory", str(Max_Steps)]
    #########################
    # ICM module
    #########################
    icm_lr = 1e-3
    ICM_hidden_dim = 128
    ICM_forward_loss_factor = 0.5
    ICM_inverse_loss_factor = 0.5
    intrinsic_reward_factor = 0.05

    # Memory_Size = 3200 #No use
    # ---------- ------------
    #Change to False if Simulation GUI must be shown
    #如果必须显示模拟GUI，则更改为False
