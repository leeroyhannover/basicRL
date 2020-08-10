# 导入基本的数据包
# ppo也是基于dqn的方式，也加入了ac的框架
import matplotlib.pyplot as plt
import sys
from collections import deque
import timeit
from datetime import timedelta
from copy import deepcopy
import numpy as np
import random
from PIL import Image
from tensorboardX import SummaryWriter # 可视化训练数据

# 导入基本的环境
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
import pybullet as p

#from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv #只有一个物体的抓取

# env1 = KukaCamGymEnv(renders=True)
# env1.render(mode='human') #
# env1.cid = p.connect(p.DIRECT) #上面是true之后显示的就已经是GUI版本了

#这里需要加入渲染的部分
env = KukaDiverseObjectEnv(renders=True, isDiscrete=False, removeHeightHack=False, maxSteps=20, numObjects=1) # 渲染True,默认的就是GUI，但是还不显示机器人
# maxsteps是一个episode里面可执行的最多的动作次数
# removeHeightHack 如果是False，每次机器人自动的做下降的动作，否则要从头开始学习
# cameraRandom 可以随机的放置相机的位置，0是位置确定，1是完全随机，大部分的多状态学习都是通过不同的相机位置 //  空间感知？
# width，Height  感知到的相机的画幅
# numObjects 盘子里面物体的数量
# isTest False用于训练，test则是用于测试的数据组
env.cid = p.connect(p.DIRECT)
action_space = spaces.Box(low=-1, high=1, shape=(5,1))

#这里插入学习网络的构造
##############
# Actor-Critic implementation 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim

#torch需要手动的选择是gpu还是cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #隐藏层的构造
def build_hidden_layer(input_dim, hidden_layers):
    """Build hidden layer.
    Params
    ======
        input_dim (int): Dimension of hidden layer input
        hidden_layers (list(int)): Dimension of hidden layers
    """
    hidden = nn.ModuleList([nn.Linear(input_dim, hidden_layers[0])])
    if len(hidden_layers)>1:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        hidden.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
    return hidden
    
    
    #构造AC过程
class ActorCritic(nn.Module):
    def __init__(self,state_size,action_size,shared_layers,
                 critic_hidden_layers=[],actor_hidden_layers=[],
                 seed=0, init_type=None):
        """Initialize parameters and build policy.
        Params
        ======
            state_size (int,int,int): Dimension of each state
            action_size (int): Dimension of each action
            shared_layers (list(int)): Dimension of the shared hidden layers
            critic_hidden_layers (list(int)): Dimension of the critic's hidden layers
            actor_hidden_layers (list(int)): Dimension of the actor's hidden layers
            seed (int): Random seed
            init_type (str): Initialization type
        """
        super(ActorCritic, self).__init__()
        self.init_type = init_type
        self.seed = torch.manual_seed(seed)
        self.sigma = nn.Parameter(torch.zeros(action_size))

        # Add shared hidden layer #3个共享的层级
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_size[0])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(state_size[1])))
        linear_input_size = convh * convw * 32
        self.shared_layers = build_hidden_layer(input_dim=linear_input_size,
                                                hidden_layers=shared_layers)
                                                
                                                
        # 构建A层和C层
        # Add critic layers
        if critic_hidden_layers:
            # Add hidden layers for critic net if critic_hidden_layers is not empty
            self.critic_hidden = build_hidden_layer(input_dim=shared_layers[-1],
                                                    hidden_layers=critic_hidden_layers)
            self.critic = nn.Linear(critic_hidden_layers[-1], 1)
        else:
            self.critic_hidden = None
            self.critic = nn.Linear(shared_layers[-1], 1)

        # Add actor layers
        if actor_hidden_layers:
            # Add hidden layers for actor net if actor_hidden_layers is not empty
            self.actor_hidden = build_hidden_layer(input_dim=shared_layers[-1],
                                                   hidden_layers=actor_hidden_layers)
            self.actor = nn.Linear(actor_hidden_layers[-1], action_size)
        else:
            self.actor_hidden = None
            self.actor = nn.Linear(shared_layers[-1], action_size)
            
        # Apply Tanh() to bound the actions
        self.tanh = nn.Tanh()  # Tanh激活函数，为啥不用relu? 激活函数就是为了展现非线性从而有强的表达能力
        
        
        # 初始化隐藏层和AC层
        # Initialize hidden and actor-critic layers
        if self.init_type is not None:
            self.shared_layers.apply(self._initialize)
            self.critic.apply(self._initialize)
            self.actor.apply(self._initialize)
            if self.critic_hidden is not None:
                self.critic_hidden.apply(self._initialize)
            if self.actor_hidden is not None:
                self.actor_hidden.apply(self._initialize)
                
    
    #初始化调用函数
    def _initialize(self, n):
        """Initialize network weights.
        """
        if isinstance(n, nn.Linear):
            if self.init_type=='xavier-uniform':
                nn.init.xavier_uniform_(n.weight.data)
            elif self.init_type=='xavier-normal':
                nn.init.xavier_normal_(n.weight.data)
            elif self.init_type=='kaiming-uniform':
                nn.init.kaiming_uniform_(n.weight.data)
            elif self.init_type=='kaiming-normal':
                nn.init.kaiming_normal_(n.weight.data)
            elif self.init_type=='orthogonal':
                nn.init.orthogonal_(n.weight.data)
            elif self.init_type=='uniform':
                nn.init.uniform_(n.weight.data)
            elif self.init_type=='normal':
                nn.init.normal_(n.weight.data)
            else:
                raise KeyError('initialization type is not found in the set of existing types')
    
    # 把state映射到action和value上面去
    def forward(self, state):
        """Build a network that maps state -> (action, value)."""
        def apply_multi_layer(layers,x,f=F.leaky_relu):
            for layer in layers:
                x = f(layer(x))
            return x

        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = F.relu(self.bn3(self.conv3(state)))
        state = apply_multi_layer(self.shared_layers,state.view(state.size(0),-1))

        v_hid = state
        if self.critic_hidden is not None:
            v_hid = apply_multi_layer(self.critic_hidden,v_hid)

        a_hid = state
        if self.actor_hidden is not None:
            a_hid = apply_multi_layer(self.actor_hidden,a_hid)

        a = self.tanh(self.actor(a_hid))
        value = self.critic(v_hid).squeeze(-1)
        return a, value  #最后返回的是动作a和value的计算结果

##############

#视觉是通过torchVision来实现的
##############
import torchvision.transforms as T

#resize在这里裁剪指定的图片大小，如果输入不是图片可以不用了
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])  #检查状态和动作空间，对输入的图片做变化，裁剪等
                    
    # 获取相机的屏幕信息
def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    # 对较大的截图做出裁剪
    #env.render(mode='human')
    # 从环境里面获得图片
    screen = env._get_observation().transpose((2, 0, 1))  # 将地二轴和第零轴的数据进行转置
    #[screen, depth, segement] = 
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255 #返回的类型要是float类型
    screen = torch.from_numpy(screen)  #从numpy获取数据转换成torch用的sensor
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)  #返回的图像返回到torch里面去


##############


# 训练的过程

env.reset() #环境需要重置

num_agents = 1  #什么意思？
print('Number of agents:', num_agents)

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape  #确定屏幕的大小

action_size = env.action_space.shape[0] #动作空间的大小，三个方向吗？由env决定的
print('Size of each action:', action_size)

    #展示第一张模拟相机图片  //  matplotlib里面的调用和matlab里面相似
plt.figure()
plt.imshow(init_screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()

    #初始化训练
writer = SummaryWriter()
i_episode = 0
ten_rewards = 0

    #定义策略计算函数
def collect_trajectories(envs, policy, tmax=200, nrand=5):
    
    #确定控制参数
    global i_episode 
    global ten_rewards
    global writer
    
    #初始化返回的参数
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]
    value_list=[]
    done_list=[]
    
    state = envs.reset() #重置环境，前面的重置只是为了取出图片
    
    #随机开始的运动模式
    # perform nrand random steps
    for _ in range(nrand):
        action = np.random.randn(action_size) #随机的运动过程
        action = np.clip(action, -1.0, 1.0) #限制action矩阵最小不小于-1最大不大于1
        _, reward, done, _  = envs.step(action) #这里把动作给进去
        reward = torch.tensor([reward], device=device)
        
        
    # 开始训练
    for t in range(tmax):
        states = get_screen()  #获取相机
        action_est, values = policy(states) #由torch产生的策略policy
        sigma = nn.Parameter(torch.zeros(action_size)) # 产生一个action_size的零矩阵
        dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(device))  #转换成正态分布
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=-1).detach()  # detach截断了反向传播的梯度流
        values = values.detach()
        actions = actions.detach()
        
        env_actions = actions.cpu().numpy()
        _, reward, done, _  = envs.step(env_actions[0])  #把动作给进去
        rewards = torch.tensor([reward], device=device)  #把reward变成tensor
        dones = torch.tensor([done], device=device)     #检测是否完成了任务

        state_list.append(states.unsqueeze(0))      #用于存储是否抓到物体
        prob_list.append(log_probs.unsqueeze(0))
        action_list.append(actions.unsqueeze(0))
        reward_list.append(rewards.unsqueeze(0))
        value_list.append(values.unsqueeze(0))
        done_list.append(dones)

        if np.any(dones.cpu().numpy()): # 每一次的训练都记得保存
            ten_rewards += reward
            i_episode += 1
            state = envs.reset()
            if i_episode%10 == 0: #抓到一次累计一次奖励，每10次写出一次奖励的值
                writer.add_scalar('ten episodes average rewards', ten_rewards/10.0, i_episode) # 这里应该是写出了奖励值的
                ten_rewards = 0

    state_list = torch.cat(state_list, dim=0)  #把当前的状态拼接上去
    prob_list = torch.cat(prob_list, dim=0)
    action_list = torch.cat(action_list, dim=0)
    reward_list = torch.cat(reward_list, dim=0)
    value_list = torch.cat(value_list, dim=0)
    done_list = torch.cat(done_list, dim=0)
    return prob_list, state_list, action_list, reward_list, value_list, done_list

    #计算返回值 // 对Q值做估测？？
def calc_returns(rewards, values, dones):
    n_step = len(rewards)
    n_agent = len(rewards[0])

    # Create empty buffer
    # 创建空的buffer用于存放数据
    GAE = torch.zeros(n_step,n_agent).float().to(device)  #GAE是啥？
    returns = torch.zeros(n_step,n_agent).float().to(device)

    # Set start values
    # 开始值的评估
    GAE_current = torch.zeros(n_agent).float().to(device)

    TAU = 0.95
    discount = 0.99
    values_next = values[-1].detach()
    returns_current = values[-1].detach()
    for irow in reversed(range(n_step)):
        values_current = values[irow]
        rewards_current = rewards[irow]
        gamma = discount * (1. - dones[irow].float())

        # Calculate TD Error 计算TD的误差
        #td_error = rewards_current + gamma * values_next - values_current  #这里也需要转换类型从long到float
        td_error = rewards_current.float() + gamma.float() * values_next.float() - values_current.float()
        # Update GAE, returns 广义优势估计，不再给出具体的截断步数，对所有的步数进行加权
        GAE_current = td_error + gamma * TAU * GAE_current
        returns_current = rewards_current.float() + gamma.float() * returns_current.float()
        # Set GAE, returns to buffer 这里是具体算法的实现过程
        GAE[irow] = GAE_current
        returns[irow] = returns_current

        values_next = values_current

    return GAE, returns

    #对policy做估测，会在下面用AC生成policy
def eval_policy(envs, policy, tmax=1000):
    reward_list=[]
    state = envs.reset()   # 这里是不是应该是states
    for t in range(tmax):
        states = get_screen() 
        action_est, values = policy(states)  # 把原有的画面作为输入放入policy里面
        sigma = nn.Parameter(torch.zeros(action_size))
        dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(device))
        actions = dist.sample()  #这里进行动作的sample
        _, reward, done, _  = envs.step(actions[0])  #这里放进env的step里面返回reward和任务是否完成
        dones = done
        reward_list.append(np.mean(reward))  # 汇报存储在列表里面，但是这是次训练里所有reward，每次的决策

        # stop if any of the trajectories is done to have retangular lists
        if np.any(dones):
            break
    return reward_list

# ## Network Architecture  构架神经网络
# 通过ppo进行更新
# ### Model update using PPO/GAE
# The hyperparameters used during training are:
# 
# Parameter | Value | Description
# ------------ | ------------- | -------------
# Number of Agents | 1 | Number of agents trained simultaneously  同时训练的智能体个数
# 
# tmax | 20 | Maximum number of steps per episode  每一步所需要的最大步数
# Epochs | 10 | Number of training epoch per batch sampling  每sample一次，training多少次
# Batch size | 128 | Size of batch taken from the accumulated  trajectories 从buffer里面拿出来的batch大小
# Discount (gamma) | 0.993 | Discount rate 折损率
# Epsilon | 0.07 | Ratio used to clip r = new_probs/old_probs during training PPO的技巧
# Gradient clip | 10.0 | Maximum gradient norm 最大的梯度正交化
# Beta | 0.01 | Entropy coefficient ？？
# Tau | 0.95 | tau coefficient in GAE  GAE要用到的方差下降
# Learning rate | 2e-4 | Learning rate  学习效率
# Optimizer | Adam | Optimization method  优化器


# run your own policy!  生成自己的policy，输入的还是原始的数据组
policy=ActorCritic(state_size=(screen_height, screen_width),
              action_size=action_size,
              shared_layers=[128, 64],
              critic_hidden_layers=[64],
              actor_hidden_layers=[64],
              init_type='xavier-uniform',
              seed=0).to(device)

# 这里用的adam优化器，同时SGD也是可以的
# we use the adam optimizer with learning rate 2e-4
# optim.SGD is also possible
optimizer = optim.Adam(policy.parameters(), lr=2e-4)

# 训练模型保存的位置
PATH = 'policy_ppo.pt'

writer = SummaryWriter()
best_mean_reward = None

scores_window = deque(maxlen=100)  # last 100 scores  生成一个双端的队列

discount = 0.993
epsilon = 0.07
beta = .01
opt_epoch = 10
season = 1000000  # 总共训练的季度数
batch_size = 128
tmax = 1000 #env episode steps，一个episode有多少个step
save_scores = []
start_time = timeit.default_timer()


#开始训练
#开始测试的时候这部分的函数不需要执行
for s in range(season):
    policy.eval() # 这个eval源代码里面并没有得以定义
    old_probs_lst, states_lst, actions_lst, rewards_lst, values_lst, dones_list = collect_trajectories(envs=env,
                                                                                                       policy=policy,
                                                                                                       tmax=tmax,
                                                                                                       nrand = 5)

    season_score = rewards_lst.sum(dim=0).item()
    scores_window.append(season_score)
    save_scores.append(season_score)
    
    gea, target_value = calc_returns(rewards = rewards_lst,
                                     values = values_lst,
                                     dones=dones_list)
    gea = (gea - gea.mean()) / (gea.std() + 1e-8)

    policy.train()

    # cat all agents
    # 所有agent的集合
    def concat_all(v):
        #print(v.shape)
        if len(v.shape) == 3:#actions
            return v.reshape([-1, v.shape[-1]])
        if len(v.shape) == 5:#states
            v = v.reshape([-1, v.shape[-3], v.shape[-2],v.shape[-1]])
            #print(v.shape)
            return v
        return v.reshape([-1])

    old_probs_lst = concat_all(old_probs_lst)
    states_lst = concat_all(states_lst)
    actions_lst = concat_all(actions_lst)
    rewards_lst = concat_all(rewards_lst)
    values_lst = concat_all(values_lst)
    gea = concat_all(gea)
    target_value = concat_all(target_value)
    
    # 梯度上升的过程
    n_sample = len(old_probs_lst)//batch_size
    idx = np.arange(len(old_probs_lst))
    np.random.shuffle(idx)
    for epoch in range(opt_epoch):
        for b in range(n_sample):
            ind = idx[b*batch_size:(b+1)*batch_size]
            g = gea[ind]
            tv = target_value[ind]
            actions = actions_lst[ind]
            old_probs = old_probs_lst[ind]

            action_est, values = policy(states_lst[ind])
            sigma = nn.Parameter(torch.zeros(action_size))
            dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(device))
            log_probs = dist.log_prob(actions)
            log_probs = torch.sum(log_probs, dim=-1)
            entropy = torch.sum(dist.entropy(), dim=-1)

            ratio = torch.exp(log_probs - old_probs)
            ratio_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            L_CLIP = torch.mean(torch.min(ratio*g, ratio_clipped*g))
            # entropy bonus
            S = entropy.mean()
            # squared-error value function loss
            L_VF = 0.5 * (tv - values).pow(2).mean()
            # clipped surrogate
            L = -(L_CLIP - L_VF + beta*S)
            optimizer.zero_grad()
            # This may need retain_graph=True on the backward pass
            # as pytorch automatically frees the computational graph after
            # the backward pass to save memory
            # Without this, the chain of derivative may get lost
            L.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
            del(L)
            
            
    # 这相当于一个标记函数，随着时间的推移标记的越来越淡
    epsilon*=.999
    
    # regulation term, this reduces exploration in later runs
    # 规范函数，也是对函数进行衰减
    beta*=.998

    # 平均奖励
    # 这里是写进去的内容，就是探索参数和衰减参数，没有给出奖励参数
    mean_reward = np.mean(scores_window) # 每100个回合计算一次平均的奖励
    writer.add_scalar("epsilon", epsilon, s)
    writer.add_scalar("beta", beta, s)

    # 只对特定的训练次数进行可视化
    if best_mean_reward is None or best_mean_reward < mean_reward:
                # For saving the model and possibly resuming training
                # 保存模型方便之后接着训练
                torch.save({
                        'policy_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epsilon': epsilon,
                        'beta': beta
                        }, PATH) # 保存模型在PATH给定的文件下面
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
    #要求season在25之上，并且平均奖励高于50则停止训练            
    if s>=25 and mean_reward>50:
        print('Environment solved in {:d} seasons!\tAverage Score: {:.2f}'.format(s+1, mean_reward))
        break


# 打印平均的得分状况
print('Average Score: {:.2f}'.format(mean_reward))
elapsed = timeit.default_timer() - start_time
print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
writer.close()
env.close()

# 这个figure是用来显示训练次数的
fig = plt.figure()
plt.plot(np.arange(len(save_scores)), save_scores)
plt.ylabel('Score')
plt.xlabel('Season #')
plt.grid()
plt.show()


# ## 评估过程

episode = 10
scores_window = deque(maxlen=100)  # last 100 scores
env = KukaDiverseObjectEnv(renders=False, isDiscrete=False, removeHeightHack=False, maxSteps=20, isTest=True)
env.cid = p.connect(p.DIRECT)
# load the model
# 前面的是先写好框架保存了再从这里读取？？
# 训练到一半的模型可以接着加载训练
checkpoint = torch.load(PATH)
policy.load_state_dict(checkpoint['policy_state_dict'])   # 涉及到了前面的很多架构

# evaluate the model
for e in range(episode):
    rewards = eval_policy(envs=env, policy=policy)
    reward = np.sum(rewards,0)
    print("Episode: {0:d}, reward: {1}".format(e+1, reward), end="\n")


