# 扩展库 Gym 的使用


```python
import numpy as np
np.random.seed(0)
import pandas as pd
import gym
```

列出所有环境


```python
space_names = ['观测空间', '动作空间', '奖励范围', '最大步数']
df = pd.DataFrame(columns=space_names)

env_specs = gym.envs.registry.all()
for env_spec in env_specs:
    env_id = env_spec.id
    try:
        env = gym.make(env_id)
        observation_space = env.observation_space
        action_space = env.action_space
        reward_range = env.reward_range
        max_episode_steps = None
        if isinstance(env, gym.wrappers.time_limit.TimeLimit):
            max_episode_steps = env._max_episode_steps
        df.loc[env_id] = [observation_space, action_space, reward_range, max_episode_steps]
    except:
        pass

with pd.option_context('display.max_rows', None):
    display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>观测空间</th>
      <th>动作空间</th>
      <th>奖励范围</th>
      <th>最大步数</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Copy-v0</th>
      <td>Discrete(6)</td>
      <td>(Discrete(2), Discrete(2), Discrete(5))</td>
      <td>(-inf, inf)</td>
      <td>200</td>
    </tr>
    <tr>
      <th>RepeatCopy-v0</th>
      <td>Discrete(6)</td>
      <td>(Discrete(2), Discrete(2), Discrete(5))</td>
      <td>(-inf, inf)</td>
      <td>200</td>
    </tr>
    <tr>
      <th>ReversedAddition-v0</th>
      <td>Discrete(4)</td>
      <td>(Discrete(4), Discrete(2), Discrete(3))</td>
      <td>(-inf, inf)</td>
      <td>200</td>
    </tr>
    <tr>
      <th>ReversedAddition3-v0</th>
      <td>Discrete(4)</td>
      <td>(Discrete(4), Discrete(2), Discrete(3))</td>
      <td>(-inf, inf)</td>
      <td>200</td>
    </tr>
    <tr>
      <th>DuplicatedInput-v0</th>
      <td>Discrete(6)</td>
      <td>(Discrete(2), Discrete(2), Discrete(5))</td>
      <td>(-inf, inf)</td>
      <td>200</td>
    </tr>
    <tr>
      <th>Reverse-v0</th>
      <td>Discrete(3)</td>
      <td>(Discrete(2), Discrete(2), Discrete(2))</td>
      <td>(-inf, inf)</td>
      <td>200</td>
    </tr>
    <tr>
      <th>CartPole-v0</th>
      <td>Box(4,)</td>
      <td>Discrete(2)</td>
      <td>(-inf, inf)</td>
      <td>200</td>
    </tr>
    <tr>
      <th>CartPole-v1</th>
      <td>Box(4,)</td>
      <td>Discrete(2)</td>
      <td>(-inf, inf)</td>
      <td>500</td>
    </tr>
    <tr>
      <th>MountainCar-v0</th>
      <td>Box(2,)</td>
      <td>Discrete(3)</td>
      <td>(-inf, inf)</td>
      <td>200</td>
    </tr>
    <tr>
      <th>MountainCarContinuous-v0</th>
      <td>Box(2,)</td>
      <td>Box(1,)</td>
      <td>(-inf, inf)</td>
      <td>999</td>
    </tr>
    <tr>
      <th>Pendulum-v0</th>
      <td>Box(3,)</td>
      <td>Box(1,)</td>
      <td>(-inf, inf)</td>
      <td>200</td>
    </tr>
    <tr>
      <th>Acrobot-v1</th>
      <td>Box(6,)</td>
      <td>Discrete(3)</td>
      <td>(-inf, inf)</td>
      <td>500</td>
    </tr>
    <tr>
      <th>Blackjack-v0</th>
      <td>(Discrete(32), Discrete(11), Discrete(2))</td>
      <td>Discrete(2)</td>
      <td>(-inf, inf)</td>
      <td>None</td>
    </tr>
    <tr>
      <th>KellyCoinflip-v0</th>
      <td>(Box(1,), Discrete(301))</td>
      <td>Discrete(25000)</td>
      <td>(0, 250.0)</td>
      <td>None</td>
    </tr>
    <tr>
      <th>KellyCoinflipGeneralized-v0</th>
      <td>(Box(1,), Discrete(309), Discrete(309), Discre...</td>
      <td>Discrete(20000)</td>
      <td>(0, 200.0)</td>
      <td>None</td>
    </tr>
    <tr>
      <th>FrozenLake-v0</th>
      <td>Discrete(16)</td>
      <td>Discrete(4)</td>
      <td>(0, 1)</td>
      <td>100</td>
    </tr>
    <tr>
      <th>FrozenLake8x8-v0</th>
      <td>Discrete(64)</td>
      <td>Discrete(4)</td>
      <td>(0, 1)</td>
      <td>200</td>
    </tr>
    <tr>
      <th>CliffWalking-v0</th>
      <td>Discrete(48)</td>
      <td>Discrete(4)</td>
      <td>(-inf, inf)</td>
      <td>None</td>
    </tr>
    <tr>
      <th>NChain-v0</th>
      <td>Discrete(5)</td>
      <td>Discrete(2)</td>
      <td>(-inf, inf)</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>Roulette-v0</th>
      <td>Discrete(1)</td>
      <td>Discrete(38)</td>
      <td>(-inf, inf)</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Taxi-v3</th>
      <td>Discrete(500)</td>
      <td>Discrete(6)</td>
      <td>(-inf, inf)</td>
      <td>200</td>
    </tr>
    <tr>
      <th>GuessingGame-v0</th>
      <td>Discrete(4)</td>
      <td>Box(1,)</td>
      <td>(-inf, inf)</td>
      <td>200</td>
    </tr>
    <tr>
      <th>HotterColder-v0</th>
      <td>Discrete(4)</td>
      <td>Box(1,)</td>
      <td>(-inf, inf)</td>
      <td>200</td>
    </tr>
    <tr>
      <th>CubeCrash-v0</th>
      <td>Box(40, 32, 3)</td>
      <td>Discrete(3)</td>
      <td>(-inf, inf)</td>
      <td>None</td>
    </tr>
    <tr>
      <th>CubeCrashSparse-v0</th>
      <td>Box(40, 32, 3)</td>
      <td>Discrete(3)</td>
      <td>(-inf, inf)</td>
      <td>None</td>
    </tr>
    <tr>
      <th>CubeCrashScreenBecomesBlack-v0</th>
      <td>Box(40, 32, 3)</td>
      <td>Discrete(3)</td>
      <td>(-inf, inf)</td>
      <td>None</td>
    </tr>
    <tr>
      <th>MemorizeDigits-v0</th>
      <td>Box(24, 32, 3)</td>
      <td>Discrete(10)</td>
      <td>(-inf, inf)</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>


# 小车上山 MountainCar-v0

环境：Gym库的 MountainCar-v0


```python
env = gym.make('MountainCar-v0')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测范围 = {} ~ {}'.format(env.observation_space.low,
        env.observation_space.high))
print('动作数 = {}'.format(env.action_space.n))
```

    观测空间 = Box(2,)
    动作空间 = Discrete(3)
    观测范围 = [-1.2  -0.07] ~ [0.6  0.07]
    动作数 = 3
    

智能体：一个根据指定确定性策略决定动作并且不学习的智能体


```python
class BespokeAgent:
    def __init__(self, env):
        pass
    
    def decide(self, observation): # 决策
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
                0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action # 返回动作

    def learn(self, *args): # 学习
        pass
    
agent = BespokeAgent(env)
```

智能体与环境交互


```python
def play_montecarlo(env, agent, render=False, train=False):
    episode_reward = 0. # 记录回合总奖励，初始化为0
    observation = env.reset() # 重置游戏环境，开始新回合
    while True: # 不断循环，直到回合结束
        if render: # 判断是否显示
            env.render() # 显示图形界面，图形界面可以用 env.close() 语句关闭
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action) # 执行动作
        episode_reward += reward # 收集回合奖励
        if train: # 判断是否训练智能体
            agent.learn(observation, action, reward, done) # 学习
        if done: # 回合结束，跳出循环
            break
        observation = next_observation
    return episode_reward # 返回回合总奖励
```

交互1回合，并图形化显示


```python
env.seed(0) # 设置随机数种子,只是为了让结果可以精确复现,一般情况下可删去
episode_reward = play_montecarlo(env, agent, render=True)
print('回合奖励 = {}'.format(episode_reward))
env.close() # 此语句可关闭图形界面
```

    回合奖励 = -105.0
    

评估性能：交互100回合求平均


```python
episode_rewards = [play_montecarlo(env, agent) for _ in range(100)]
print('平均回合奖励 = {}'.format(np.mean(episode_rewards)))
```

    平均回合奖励 = -102.61
    
