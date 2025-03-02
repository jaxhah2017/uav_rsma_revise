import random
import collections
import numpy as np

action_key = {'moves', 'powers', 'thetas'}

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, actions, reward, next_state, done):  # 将数据加入buffer
        n_agent = len(state)

        action = []

        done = np.full(n_agent, done)

        for act in actions:
            aa = []
            aa.extend(act['moves'])
            aa.extend(act['powers'])
            aa.extend(act['thetas'])
            action.append(aa)
        

        self.buffer.append((state, action, reward, next_state, done))
        

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)