import torch
import torch.nn as nn
import math
    
import torch.nn.functional as F

class Agents_actor(nn.Module):
    def __init__(self, input_dim, move_dim, power_dim, theta_dim, n_layers=2, hidden_size=256):
        super(Agents_actor, self).__init__()
        self._n_layers = n_layers
        self._hidden_size = hidden_size
        
        layers = [nn.Linear(input_dim, self._hidden_size), nn.ReLU()]
        for l in range(self._n_layers - 1):
            layers += [nn.Linear(self._hidden_size, self._hidden_size), nn.ReLU()]
        self.enc = nn.Sequential(*layers)
        self.rnn = nn.GRUCell(self._hidden_size, self._hidden_size)

        self.move_head = nn.Linear(self._hidden_size, move_dim)
        self.power_head = nn.Linear(self._hidden_size, power_dim)
        self.theta_head = nn.Linear(self._hidden_size, theta_dim) # test

    def init_hidden(self):
        return torch.zeros(1, self._hidden_size)

    def forward(self, x, h):
        x = self.enc(x)

        h = self.rnn(x, h)

        move = self.move_head(h)
        power = self.power_head(h)
        theta = self.theta_head(h)

        
        return move, power, theta, h
    
class Agents_critic(nn.Module):
    def __init__(self, input_dim, n_layers=2, hidden_size=256):
        super(Agents_critic, self).__init__()
        self._n_layers = n_layers
        self._hidden_size = hidden_size
        
        layers = [nn.Linear(input_dim, self._hidden_size), nn.ReLU()]
        for l in range(self._n_layers - 1):
            layers += [nn.Linear(self._hidden_size, self._hidden_size), nn.ReLU()]
        self.enc = nn.Sequential(*layers)
        self.rnn = nn.GRUCell(self._hidden_size, self._hidden_size)
        self.q = nn.Linear(self._hidden_size, 1)

    def init_hidden(self):
        return torch.zeros(1, self._hidden_size)

    def forward(self, x, h):
        x = self.enc(x)

        h = self.rnn(x, h)

        q_value = self.q(h)

        return q_value, h

if __name__ == '__main__':
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    batch_size = 2
    num_gts = 12
    gt_features_dim = 4  # 每个GT的特征维度
    num_heads = 2
    other_dim = 6
    moves_dim = 16
    jamming_power_dim = 10
    hidden_dim = 256

    agent = Agents_actor(gt_features_dim=gt_features_dim,
                   num_heads=num_heads,
                   other_features_dim=other_dim,
                   move_dim=moves_dim,
                   power_dim=jamming_power_dim)
    gt_features = torch.zeros(batch_size, num_gts, gt_features_dim)
    other_features = torch.rand(batch_size, other_dim)
    hidden_state = agent.init_hidden().expand(batch_size, -1)  # 扩展隐藏状态到批大小

    # 模型向前传播
    output = agent(gt_features, other_features, hidden_state)
    print(output[2])


