import matplotlib.pyplot as plt

import numpy as np

def plot2uav_traj(init_info, uav_traj, jamming_power, args, pic_path):
    uav_init_pos = init_info['uav_init_pos']
    eve_init_pos = init_info['eve_init_pos']
    gts_init_pos = init_info['gts_init_pos']
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].axis([0, 400, 0, 400])
    for (x, y) in uav_init_pos:
        ubs_init, = ax[0].plot(x, y, marker='o', color='b')
    for (x, y) in eve_init_pos:
        eve_init, = ax[0].plot(x, y, marker='o', color='r', markersize=5)
    for (x, y) in gts_init_pos:
        gts_init, = ax[0].plot(x, y, marker='o', color='y', markersize=5)
    # plot axis0
    for x in range(0, 400, int(100)):
        if x == 0:
            continue
        ax[0].plot([0, 400], [x, x], linestyle='--', color='b')

    # plot axis1
    for y in range(0, 400, int(100)):
        if y == 0:
            continue
        ax[0].plot([y, y], [0, 400], linestyle='--', color='b')
    # uav1_pos, uav2_pos, uav3_pos, uav4_pos = uav_init_pos[0], uav_init_pos[1], uav_init_pos[2], uav_init_pos[3]
    uav1_pos, uav2_pos = uav_init_pos[0], uav_init_pos[1]
    for i, traj in enumerate(uav_traj):
        uav1_pos_new = traj[0]
        uav2_pos_new = traj[1]
        ubs1_trajectory, = ax[0].plot([uav1_pos[0], uav1_pos_new[0]],
                                      [uav1_pos[1], uav1_pos_new[1]], linestyle='-', color='c', linewidth=2)
        ubs2_trajectory, = ax[0].plot([uav2_pos[0], uav2_pos_new[0]],
                                      [uav2_pos[1], uav2_pos_new[1]], linestyle='-', color='g', linewidth=2)
        uav1_pos = uav1_pos_new
        uav2_pos = uav2_pos_new

    uav1_final_pos, = ax[0].plot(uav1_pos[0], uav1_pos[1], marker='o', color='c')
    uav2_final_pos, = ax[0].plot(uav2_pos[0], uav2_pos[1], marker='o', color='g')
    ax[0].legend(handles=[ubs_init, eve_init, gts_init,
                          ubs1_trajectory, ubs2_trajectory,
                          uav1_final_pos, uav2_final_pos],
                 labels=['uav_init_pos', 'eve_pos', 'gts_pos',
                         'uav1_traj', 'uav2_traj',
                         'uav1_final_pos', 'uav2_final_pos'],
                 loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax[0].set_title("UAV trajectory")
    ax[0].set_xlabel("X (m)")
    ax[0].set_ylabel("Y (m)")
    fig.subplots_adjust(wspace=0.5)

    x = np.array(range(0, args.episode_length - 1))
    uav_jamming_power = np.array(jamming_power, dtype=np.float32)
    uav0_jamming_power = uav_jamming_power[:, 0]
    uav1_jamming_power = uav_jamming_power[:, 1]
    ax[1].plot(x, uav0_jamming_power, linestyle='-', color='c', linewidth=2, label='uav0 j_power')
    ax[1].plot(x, uav1_jamming_power, linestyle='-', color='g', linewidth=2, label='uav1 j_power')
    ax[1].legend(loc="center left",
                 bbox_to_anchor=(1, 0.5))
    ax[1].set_title("UAV jamming power (Watt)")
    ax[1].set_xlabel("Time steps")
    ax[1].set_ylabel("Jamming power")
    plt.savefig(pic_path + '/test2uav.png')


def plot4uav_traj(init_info, uav_traj, jamming_power, args, pic_path):
    uav_init_pos = init_info['uav_init_pos']
    eve_init_pos = init_info['eve_init_pos']
    gts_init_pos = init_info['gts_init_pos']

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].axis([0, 400, 0, 400])
    for (x, y) in uav_init_pos:
        ubs_init, = ax[0].plot(x, y, marker='o', color='b')
    for (x, y) in eve_init_pos:
        eve_init, = ax[0].plot(x, y, marker='o', color='r', markersize=5)
    for (x, y) in gts_init_pos:
        gts_init, = ax[0].plot(x, y, marker='o', color='y', markersize=5)
    # plot axis0
    for x in range(0, 400, int(100)):
        if x == 0:
            continue
        ax[0].plot([0, 400], [x, x], linestyle='--', color='b')

    # plot axis1
    for y in range(0, 400, int(100)):
        if y == 0:
            continue
        ax[0].plot([y, y], [0, 400], linestyle='--', color='b')
    uav1_pos, uav2_pos, uav3_pos, uav4_pos = uav_init_pos[0], uav_init_pos[1], uav_init_pos[2], uav_init_pos[3]
    # uav1_pos, uav2_pos = uav_init_pos[0], uav_init_pos[1]
    for i, traj in enumerate(uav_traj):
        uav1_pos_new = traj[0]
        uav2_pos_new = traj[1]
        uav3_pos_new = traj[2]
        uav4_pos_new = traj[3]
        # print("plot:", uav1_pos_new, uav2_pos_new)
        ubs1_trajectory, = ax[0].plot([uav1_pos[0], uav1_pos_new[0]],
                                      [uav1_pos[1], uav1_pos_new[1]], linestyle='-', color='#7158e2', linewidth=2)
        ubs2_trajectory, = ax[0].plot([uav2_pos[0], uav2_pos_new[0]],
                                      [uav2_pos[1], uav2_pos_new[1]], linestyle='-', color='#3ae374', linewidth=2)
        ubs3_trajectory, = ax[0].plot([uav3_pos[0], uav3_pos_new[0]],
                                   [uav3_pos[1], uav3_pos_new[1]], linestyle='-', color='#3d3d3d', linewidth=2)
        ubs4_trajectory, = ax[0].plot([uav4_pos[0], uav4_pos_new[0]],
                                   [uav4_pos[1], uav4_pos_new[1]], linestyle='-', color='#ffb8b8', linewidth=2)
        uav1_pos = uav1_pos_new
        uav2_pos = uav2_pos_new
        uav3_pos = uav3_pos_new
        uav4_pos = uav4_pos_new

    uav1_final_pos, = ax[0].plot(uav1_pos[0], uav1_pos[1], marker='o', color='#7158e2')
    uav2_final_pos, = ax[0].plot(uav2_pos[0], uav2_pos[1], marker='o', color='#3ae374')
    uav3_final_pos, = ax[0].plot(uav3_pos[0], uav3_pos[1], marker='o', color='#3d3d3d')
    uav4_final_pos, = ax[0].plot(uav4_pos[0], uav4_pos[1], marker='o', color='#ffb8b8')
    ax[0].legend(handles=[ubs_init, eve_init, gts_init,
                       ubs1_trajectory, ubs2_trajectory,
                       ubs3_trajectory, ubs4_trajectory,
                       uav1_final_pos, uav2_final_pos,
                       uav3_final_pos, uav4_final_pos],
              labels=['uav_init_pos', 'eve_pos', 'gts_pos',
                      'uav1_traj', 'uav2_traj',
                      'uav3_traj', 'uav4_traj',
                      'uav1_final_pos', 'uav2_final_pos',
                      'uav3_final_pos', 'uav4_final_pos'],
              loc="center left", bbox_to_anchor=(1, 0.5))
    # ax[0].legend(handles=[ubs_init, eve_init, gts_init,
    #                       ubs1_trajectory, ubs2_trajectory,
    #                       uav1_final_pos, uav2_final_pos],
    #              labels=['uav_init_pos', 'eve_pos', 'gts_pos',
    #                      'uav1_traj', 'uav2_traj',
    #                      'uav1_final_pos', 'uav2_final_pos'],
    #              loc="center left", bbox_to_anchor=(1, 0.5))
    ax[0].set_title("UAV trajectory")
    ax[0].set_xlabel("X (m)")
    ax[0].set_ylabel("Y (m)")
    fig.subplots_adjust(wspace=0.5)

    x = np.array(range(0, args.episode_length - 1))
    uav_jamming_power = np.array(jamming_power, dtype=np.float32)
    uav0_jamming_power = uav_jamming_power[:, 0]
    uav1_jamming_power = uav_jamming_power[:, 1]
    uav2_jamming_power = uav_jamming_power[:, 2]
    uav3_jamming_power = uav_jamming_power[:, 3]
    ax[1].plot(x, uav0_jamming_power, linestyle='-', color='#7158e2', linewidth=2, label='uav0 j_power')
    ax[1].plot(x, uav1_jamming_power, linestyle='-', color='#3ae374', linewidth=2, label='uav1 j_power')
    ax[1].plot(x, uav2_jamming_power, linestyle='-', color='#3d3d3d', linewidth=2, label='uav2 j_power')
    ax[1].plot(x, uav3_jamming_power, linestyle='-', color='#ffb8b8', linewidth=2, label='uav3 j_power')
    ax[1].legend(loc="center left",
                 bbox_to_anchor=(1, 0.5))
    ax[1].set_title("UAV jamming power (Watt)")
    ax[1].set_xlabel("Time steps")
    ax[1].set_ylabel("Jamming power")
    # plt.show()
    plt.savefig(pic_path + '/test4uav.png')

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

def plot_test_returns(test_returns, pic_save_path, smooth: bool = False, smooth_beta: float = 0.95):
    returns = np.array(test_returns)
    return_means = returns.mean(axis=1)
    return_stds = returns.std(axis=1)

    if smooth:
        return_means = exp_weight_moving_average(return_means, beta=smooth_beta)
        return_stds = exp_weight_moving_average(return_stds, beta=smooth_beta)

    return_means = np.array(return_means)
    return_stds = np.array(return_stds)

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

    plt.savefig(pic_save_path + '/test_Returns4uav.png', dpi=600)

def plot_fair_index(epi_fair_index, pic_save_path):
    # print(epi_fair_index)
    plt.cla()
    plt.figure(figsize=(7, 6))
    episode_length = len(epi_fair_index)
    indices = np.array(range(0, episode_length))
    plt.plot(indices, epi_fair_index, label='Fairness', color='#E84393')
    plt.legend(loc="best")
    plt.title("Fairness Index")
    plt.xlabel("Time Step")
    plt.ylabel("Fairness Index")
    plt.xlim((0, len(epi_fair_index) + 1))
    plt.ylim((0, 1))

    plt.savefig(pic_save_path + '/epi_fair_index.png', dpi=600)

def plot_secrecy_rate(epi_secrecy_rate, pic_save_path):
    plt.cla()
    plt.figure(figsize=(7, 6))
    episode_length = len(epi_secrecy_rate)
    indices = np.array(range(0, episode_length))
    plt.plot(indices, epi_secrecy_rate, label='Secrecy Rate', color='#E84393')
    plt.legend(loc="best")
    plt.title("Secrecy Rate")
    plt.xlabel("Time Step")
    plt.ylabel("Secrecy Rate (Mbits)")
    plt.xlim((0, len(epi_secrecy_rate) + 1))

    plt.savefig(pic_save_path + '/epi_secrecy_rate.png', dpi=600)

def plot_throughput(epi_throughput, pic_save_path):
    plt.cla()
    plt.figure(figsize=(7, 6))
    episode_length = len(epi_throughput)
    indices = np.array(range(0, episode_length))
    plt.plot(indices, epi_throughput, label='Throughput', color='#E84393')
    plt.legend(loc="best")
    plt.title("Throughput")
    plt.xlabel("Time Step")
    plt.ylabel("Throughput (Mbits)")
    plt.xlim((0, len(epi_throughput) + 1))

    plt.savefig(pic_save_path + '/epi_throughput.png', dpi=600)

def plot_opt_theta(epi_theta, pic_save_path):
    color = ['r', 'g', 'b', 'y']
    plt.cla()
    plt.figure(figsize=(7, 6))
    episode_length = len(epi_theta)
    epi_theta = np.array(epi_theta)
    indices = np.array(range(0, episode_length))
    print(len(epi_theta))
    for _ in range(len(epi_theta[0])):
        plt.plot(indices, epi_theta[:, _], label=f'UAV {_}', color=color[_])
    plt.legend(loc="best")
    plt.title("Optimal Theta")
    plt.xlabel("Time Step")
    plt.ylabel("Optimal Theta")
    plt.xlim((0, len(epi_theta) + 1))

    plt.savefig(pic_save_path + '/epi_theta.png', dpi=600)