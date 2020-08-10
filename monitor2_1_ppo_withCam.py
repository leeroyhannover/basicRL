
# 检查callback进行可视化
import os

# 环境的包
import gym
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv




# 构架模型的包
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv  # 这个可能是个wrapper,处理参数之间的统一性问题
from stable_baselines import PPO2

# 用于实时显示的代码
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
#from stable_baselines.common.noise import AdaptiveParamNoiseSpec  # 增加模型的噪声
from stable_baselines.common.callbacks import BaseCallback   #这里导入callback的基础函数



# 实例化callback的类别
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

# 保存路径
log_dir = "./ppo2_kukaWithCam_Callback/"
os.makedirs(log_dir, exist_ok=True)

# 导入环境
env = KukaCamGymEnv(renders=True, isDiscrete=True)  # pybullet可以直接make
env.cid = p.connect(p.DIRECT)
env = Monitor(env, log_dir)

# 添加基本的噪声参数
# # Add some param noise for exploration
# param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
# # Because we use parameter noise, we should use a MlpPolicy with layer normalization
# model = DDPG(LnMlpPolicy, env, param_noise=param_noise, verbose=0)

# 设置超参
model = PPO2(MlpPolicy, env, verbose=1,tensorboard_log="./ppo2_kukaWithCam_tboard/") # verbose应该是一个决定运行状态的参数

# 每1000步检查一次callback
# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# 开始训练
time_steps = 1e8
model.learn(total_timesteps=int(time_steps), callback=callback)

env.close()
p.disconnect()

# 实时的绘制训练结果
results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "PPO2 Kuka no Cam")
plt.show()
