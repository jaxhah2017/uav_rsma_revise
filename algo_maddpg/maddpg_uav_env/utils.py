import numpy as np

from scipy import integrate

import torch

# import multi_uav_env.maps as maps

import random

# def regis_map(range_pos):
#     if range_pos == 400:
#         return maps.Rang400MapSpecial()

#     return None

def wrapper_obs(obs):
    observation = []
    for o in obs:
        o_agent = list(o['agent'])
        o_other_ubs = []
        for ubs in o['ubs']:
            o_other_ubs += list(ubs)
        o_gt = []
        for gt in o['gt']:
            o_gt += list(gt)
        observation.append(o_agent + o_other_ubs + o_gt)

    return observation
    # return torch.tensor(observation)

def compute_jain_fairness_index(x):
    """Computes the Jain's fairness index of entries in given ndarray."""
    if x.size > 0:
        x = np.clip(x, 1e-6, np.inf)
        return np.square(x.sum()) / (x.size * np.square(x).sum())
    else:
        return 1

def wrapper_state(state):
    s_tensor = torch.tensor(state)

    return s_tensor.unsqueeze(0)

def set_rand_seed(seed=10):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == '__main__':
    pass