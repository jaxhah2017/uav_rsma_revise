import random
import collections
import numpy as np

action_key = {'moves', 'powers', 'thetas'}

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, actions, reward, next_state, done, hs, h2s):  # 将数据加入buffer
        n_agent = len(state)

        action = []

        done = np.full(n_agent, done)

        for act in actions:
            aa = []
            aa.extend(act['moves'])
            aa.extend(act['powers'])
            aa.extend(act['thetas'])
            action.append(aa)
        
        hss = []
        for h in hs:
            h = h.tolist()[0]
            hss.append(h)
        h2ss = []
        for h2 in h2s:
            h2 = h2.tolist()[0]
            h2ss.append(h2)


        self.buffer.append((state, action, reward, next_state, done, hss, h2ss))
        

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, hs, h2s = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done, hs, h2s

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)