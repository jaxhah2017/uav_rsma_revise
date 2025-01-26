import numpy as np

import matplotlib.pyplot as plt

from algo_mha_drqn.utils import *

def exp_weight_moving_average(data, beta=0.95):
    smoothed_data = []
    current_value = 0

    for i, x in enumerate(data):
        if i == 0:
            current_value = x
        else:
            current_value = beta * current_value + (1 - beta) * x
        smoothed_data.append(current_value)

    return smoothed_data

def plot_testReturns_cmp(test_returns):
    returns = np.array(test_returns)
    return_means = returns.mean(axis=1)
    return_stds = returns.std(axis=1)

    return_means = exp_weight_moving_average(return_means)
    return_stds = exp_weight_moving_average(return_stds)

    return_means = np.array(return_means)
    return_stds = np.array(return_stds)

    # return_means = return_means[:100]
    # return_stds = return_stds[:100]
    # returns_indices = np.arange(1, 101)
    returns_indices = np.arange(len(return_means))
    plt.plot(returns_indices, return_means, label='FSF', color='r')
    plt.fill_between(returns_indices, return_means - return_stds, return_means + return_stds,
                     color='r', alpha=0.25)

    plt.tick_params(axis='both', which='major', length=7, width=2, color='#3B3B98')
    plt.tick_params(axis='both', which='minor', length=4, width=2, color='#3B3B98')
    plt.xlim((0, len(return_means) + 1))
    plt.xlabel('Episode')
    plt.ylabel('Test Return')

    # plt.xticks(np.arange(0, len(return_ltf_means), 50))
    plt.legend()

    plt.savefig('testReturnsComp_4uav.png', dpi=600)

#     plt.show()


if __name__ == '__main__':
    test_returns_path = './test_returns'
    test_returns = load_var(test_returns_path)
    plot_testReturns_cmp(test_returns)