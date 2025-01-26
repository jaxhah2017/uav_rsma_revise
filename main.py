from multi_uav_env.maps import General4uavMap, General2uavMap
from multi_uav_env.multi_ubs_env import MultiUbsRsmaEvn

import argparse

import numpy as np

import torch

from algo_mha_drqn.utils import *
from algo_mha_drqn.ma_learner import MultiAgentQLearner

from tensorboardX import SummaryWriter

def train(args, train_kwards: dict = dict()):
    for i in range(1, 100):
        output_dir = './mha_drqn_data/exp'
        if not os.path.exists(output_dir + str(i)):
            output_dir = './mha_drqn_data/exp'
            output_dir = output_dir + str(i)
            os.makedirs(output_dir)
            os.makedirs(output_dir + '/checkpoints')
            os.makedirs(output_dir + '/logs')
            os.makedirs(output_dir + '/vars')
            break
    
    # 参数设置
    args.__dict__.update(train_kwards)
    args.output_dir = output_dir
    args = check_args_sanity(args)
    save_config(output_dir=args.output_dir, config=args)  #保存设置文件

    # 设置随机数种子
    set_rand_seed(args.seed)

    # 初始化环境
    env = MultiUbsRsmaEvn(args)
    test_env = MultiUbsRsmaEvn(args)
    env_info = env.get_env_info()

    learner = MultiAgentQLearner(env_info, args)
    total_steps = args.steps_per_epoch * args.epochs
    update_after = max(args.update_after, learner.batch_size * learner.max_seq_len)  # Number of steps before updates
    update_every = learner.max_seq_len  # 模型更新间隔

    # 设置探索策略
    eps_start, eps_end = 1, 0.05  # 初始化探索率
    eps_thres = lambda t: max(eps_end, -(eps_start - eps_end) / args.decay_steps * t + eps_start)  # Epsilon scheduler

    test_agents = 0
    test_p_ret = []
    ep_ret_list = []
    def test_agent(test_agents, num_test_episodes):
        with torch.no_grad():
            """Tests the performance of trained agents."""
            returns_mean = []
            returns = np.zeros(env_info['n_ubs'], dtype=np.float32)
            for n in range(args.num_test_episodes):
                reward = 0
                (o, _, init_info), h, d = test_env.reset(), learner.init_hidden(), False  # Reset drqn_env and RNN.
                while not d:  # one episode
                    a, h, inference_time = learner.take_actions(o, h, 0.05)  # Take (quasi) deterministic actions at test time.
                    o, _, _, d, info = test_env.step(a)  # Env step
                returns += info["EpRet"]
                returns_mean.append(info["EpRet"].mean())
            returns /= num_test_episodes
            for agt in range(env_info['n_ubs']):
                writer.add_scalar("evaluate returns/agent{} ep_ret".format(agt), returns[agt], test_agents)

            return returns_mean

    # 开始训练
    episode = 0
    updates = 0
    (o, s, init_info), h = env.reset(), learner.init_hidden()  # Reset drqn_env and RNN hidden states.
    writer = SummaryWriter(log_dir=args.output_dir + '/logs')
    for t in range(total_steps):
        # Select actions following epsilon-greedy strategy.
        a, h2, inference_time = learner.take_actions(o, h, eps_thres(t))
        # Call environment step.
        o2, s2, r, d, info = env.step(a)
        # Store experience to replay buffer.
        learner.cache(o, h, s, a, r, o2, h2, s2, d, info.get("BadMask"))
        # Move to next timestep.
        o, s, h = o2, s2, h2
        # Reach the end of an episode.
        if d:
            episode += 1  # On episode completes.
            ep_ret_list.append(info["EpRet"])
            writer.add_scalar("train environment/avg fairness index",
                              info["avg_fair_idx_per_episode"] / args.episode_length, episode)
            writer.add_scalar("train environment/total throughput", info["total_throughput"], episode)
            writer.add_scalar("train returns/mean returns", info["mean_returns"], episode)
            for agt in range(env_info['n_ubs']):
                writer.add_scalar("train returns/agent{} ep_ret".format(agt), info["EpRet"][agt], episode)
            print(
                "智能体与环境交互第{}次, ep_ret = {}, total_throughput={}, average fair_idx = {}, ssr_system_rate = {}".
                format(
                    episode,
                    info['EpRet'],
                    info["total_throughput"],
                    info['avg_fair_idx_per_episode'] / args.episode_length,
                    info['Ssr_Sys']))
            (o, s, init_info), h = env.reset(), learner.init_hidden()  # Reset drqn_env and RNN hidden states.
        if (t >= update_after) and (t % update_every == 0):
            # print("--------------------learner update--------------------")
            updates += 1
            diagnostic = learner.update()
            # End of epoch handling
        if (t + 1) % args.steps_per_epoch == 0:
            epoch = (t + 1) // args.steps_per_epoch
            # Test performance of trained agents.
            returns_mean = test_agent(test_agents, args.num_test_episodes)
            test_p_ret.append(returns_mean)
            test_agents += 1
            # Anneal learning rate.
            if learner.anneal_lr:
                learner.lr_scheduler.step()
            # Save model parameters.
            if (epoch % args.save_freq == 0) or (epoch == args.epochs):
                save_path = args.output_dir + '/checkpoints/checkpoint_epoch{}.pt'.format(epoch)
                learner.save_model(save_path, stamp=dict(epoch=epoch, t=t))
                save_var(var_path=args.output_dir + '/vars/test_p_ret', var=test_p_ret)
                save_var(var_path=args.output_dir + '/vars/ep_ret_list', var=ep_ret_list)
    writer.close()
    print("Complete.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # 训练设置
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    parser.add_argument('--cuda_deterministic', type=bool, default=False, help='cuda_deterministic')
    parser.add_argument('--cuda_index', type=int, default=0, help='Cuda Index')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')

    # 智能体设置
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers of agent')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of n_heads of agent')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size of agent')

    # 算法设置
    parser.add_argument('--share_reward', type=bool, default=False, help='share reward')
    parser.add_argument('--steps_per_epoch', type=int, default=30000, help='steps per epoch')
    parser.add_argument('--update_after', type=int, default=20000, help='update after')
    parser.add_argument('--polyak', type=float, default=0.999, help='Interpolation factor in polyak averaging for target network')
    parser.add_argument('--double_q', type=bool, default=True, help='double q')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.01, help='Target soft update rate')
    parser.add_argument('--replay_size', type=int, default=int(1e4), help='Capacity of replay buffer')
    parser.add_argument('--decay_steps', type=int, default=int(5e4), help='Number of timesteps for exploration')
    parser.add_argument('--max_seq_len', default=None, help='Maximum length of time sequence')
    parser.add_argument('--mixer', type=bool, default=False, help='Whether to use mixer')
    parser.add_argument('--anneal_lr', type=bool, default=False, help='Whether to anneal learning rate')
    parser.add_argument('--episode_length', type=int, default=100, help='Episode length')
    parser.add_argument('--num_test_episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency')

    # 交互环境设置
    parser.add_argument('--map', default=General4uavMap, help='map type')
    parser.add_argument('--apply_small_fading', type=bool, default=False)
    parser.add_argument('--cov_range', type=float, default=50, help='coverage range (m)')
    parser.add_argument('--comm_range', type=float, default=np.inf, help='communication range (m)')
    parser.add_argument('--serv_capacity', type=int, default=5, help='service capacity')
    parser.add_argument('--velocity_bound', type=float, default=20, help='velocity of UAV (m/s)')
    parser.add_argument('--jamming_power_bound', type=float, default=15, help='jamming power of UAV (dBm)')
    parser.add_argument('--fair_service', type=bool, default=True, help='fair service')
    parser.add_argument('--n_powers', type=int, default=10, help='the number of power level')
    parser.add_argument('--n_dirs', type=int, default=16, help='the number of move directions')
    parser.add_argument('--avoid_collision', type=bool, default=False, help='Whether to calculate collision penalty')
    parser.add_argument('--penlty', type=float, default=0.2, help='collision penlty')

    args = parser.parse_args()

    train_kwargs ={}

    # train_kwargs = {"map": General2uavMap}

    train(args, train_kwargs)

