import torch

from buffer import ReplayBuffer

from maddpg_uav_env.multi_ubs_env import MultiUbsRsmaEvn

from utils import *

from maddpg import MADDPG

import os

from maddpg_uav_env.maps import General4uavMap, General2uavMap

from tensorboardX import SummaryWriter

def train(args, train_kwards: dict = dict(), expname=''):
    output_dir = '/home/zlj/uav_rsma_revise/algo_maddpg/maddpg_data/' + expname + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir + '/checkpoints')
        os.makedirs(output_dir + '/logs')
        os.makedirs(output_dir + '/vars')

    # 参数设置
    args.__dict__.update(train_kwards)
    args.output_dir = output_dir
    args = check_args_sanity(args)
    save_config(output_dir=args.output_dir, config=args)  #保存设置文件

    # 设置随机数种子
    set_rand_seed(args.seed)
    
    env = MultiUbsRsmaEvn(args)
    test_env = MultiUbsRsmaEvn(args)
    env_info = env.get_env_info()
    replay_buffer = ReplayBuffer(args.replay_size)
    learner = MADDPG(env_info=env_info, args=args)
    update_after = max(args.update_after, args.batch_size * args.episode_length) 
    update_every = args.episode_length * 2

    test_agents = 0
    test_p_ret = []
    ep_ret_list = []
    def evaluate(num_test_episodes=10, test_agents=1):
        with torch.no_grad():
            """Tests the performance of trained agents."""
            returns_mean = []
            returns = np.zeros(env_info['n_ubs'], dtype=np.float32)
            for n in range(args.num_test_episodes):
                reward = 0
                (o, _, init_info), d = test_env.reset(), False  # Reset drqn_env and RNN.
                h = learner.init_hidden()
                while not d:  # one episode
                    a, h = learner.take_action(o, h, explore=False)  # Take (quasi) deterministic actions at test time.
                    o, _, _, d, info = test_env.step(a)  # Env step
                returns += info["EpRet"]
                returns_mean.append(info["EpRet"].mean())
            returns /= num_test_episodes
            for agt in range(env_info['n_ubs']):
                writer.add_scalar("evaluate returns/agent{} ep_ret".format(agt), returns[agt], test_agents)

            return returns_mean
    
    return_list = [] 
    total_step = 0
    writer = SummaryWriter(log_dir=args.output_dir + '/logs')
    epoch = 0
    actor_update = 0
    h = learner.init_hidden()
    for i_episode in range(args.num_episodes):
        o, s, init_info = env.reset()
        for e_i in range(args.episode_length):
            a, h2 = learner.take_action(o, h, explore=True)
            o2, s2, r, d, info = env.step(a)
            replay_buffer.add(o, a, r, o2, d, h, h2)
            o, s = o2, s2
            
            total_step += 1

            if replay_buffer.size(
            ) >= args.minimal_size and (total_step 
                                        >= update_after
                                        ) and (total_step % update_every == 0):
                sample = replay_buffer.sample(args.batch_size)

                def stack_array(x):
                    rearranged = [[sub_x[i] for sub_x in x]
                                for i in range(len(x[0]))]
                    return [
                        torch.FloatTensor(np.vstack(aa)).to(args.device)
                        for aa in rearranged
                    ]

                sample = [stack_array(x) for x in sample]
                for a_i in range(env_info['n_ubs']):
                    learner.update(sample, a_i)
                actor_update += 1
                if actor_update % 5 == 0:
                    learner.update_all_targets()
                    actor_update = 0
            
            if d:
                ep_ret_list.append(info["EpRet"])
                writer.add_scalar("train environment/avg fairness index",
                              info["avg_fair_idx_per_episode"] / args.episode_length, i_episode)
                writer.add_scalar("train environment/total throughput", info["total_throughput"], i_episode)
                writer.add_scalar("train returns/mean returns", info["mean_returns"], i_episode)
                for agt in range(env_info['n_ubs']):
                    writer.add_scalar("train returns/agent{} ep_ret".format(agt), info["EpRet"][agt], i_episode)
                print(
                    "智能体与环境交互第{}次, ep_ret = {}, total_throughput={}, average fair_idx = {}, ssr_system_rate = {}".
                    format(
                        i_episode,
                        info['EpRet'],
                        info["total_throughput"],
                        info['avg_fair_idx_per_episode'] / args.episode_length,
                        info['Ssr_Sys']))

            if total_step % args.test_per_steps == 0 and total_step != 0:
                epoch += 1
                return_means = evaluate(num_test_episodes=10, test_agents=epoch)
                test_p_ret.append(return_means)
                if args.anneal_lr:
                    learner.lr_step()
                save_path = args.output_dir + '/checkpoints/checkpoint_epoch{}.pt'.format(epoch)
                if epoch % 10 == 0:
                    learner.save_model(save_path, stamp=dict(epoch=epoch, t=total_step))
                    save_var(var_path=args.output_dir + '/vars/test_p_ret', var=test_p_ret)
                    save_var(var_path=args.output_dir + '/vars/ep_ret_list', var=ep_ret_list)
    writer.close()
    print("Complete.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # 训练设置
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    parser.add_argument('--cuda_deterministic', type=bool, default=False, help='cuda_deterministic')
    parser.add_argument('--cuda_index', type=int, default=0, help='Cuda Index')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--num_episodes', type=int, default=60000, help='num_episodes')
    parser.add_argument('--update_interval', type=int, default=100, help='update_interval')
    parser.add_argument('--minimal_size', type=int, default=4000, help='minimal_size') 

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
    parser.add_argument('--test_per_steps', type=int, default=200, help='Save frequency')
    parser.add_argument('--theta_opt', type=bool, default=False, help='Whether to do theta optimize')

    # 交互环境设置
    parser.add_argument('--map', default='General4uavMap', help='map type')
    parser.add_argument('--apply_small_fading', type=bool, default=False)
    parser.add_argument('--cov_range', type=float, default=50, help='coverage range (m)')
    parser.add_argument('--comm_range', type=float, default=np.inf, help='communication range (m)')
    parser.add_argument('--serv_capacity', type=int, default=5, help='service capacity')
    parser.add_argument('--velocity_bound', type=float, default=20, help='velocity of UAV (m/s)')
    parser.add_argument('--jamming_power_bound', type=float, default=15, help='jamming power of UAV (dBm)')
    parser.add_argument('--fair_service', type=bool, default=True, help='fair service')
    parser.add_argument('--n_powers', type=int, default=10, help='the number of power level')
    parser.add_argument('--n_dirs', type=int, default=16, help='the number of move directions') # 修改了方向个数
    parser.add_argument('--avoid_collision', type=bool, default=True, help='Whether to calculate collision penalty')
    parser.add_argument('--penlty', type=float, default=0.2, help='collision penlty')
    parser.add_argument('--sigma_err_sq', type=float, default=0.1, help='Channel Estimation Error')
    parser.add_argument('--apply_err_sq', type=bool, default=False, help='Application channel estimation error')
    parser.add_argument('--trans_scheme', type=str, default='Proposed', help='Transmission scheme')
    parser.add_argument('--algo', type=str, default='Proposed', help='Algorithm used')


    args = parser.parse_args()

    map_otions = {'General4uavMap': 'General4uavMap',
                  'General2uavMap': 'General2uavMap'}

    transmission_scheme_options = {'Proposed': 'Proposed',
                                   'RSMA': 'RSMA',
                                   'C_NOMA': 'C_NOMA'}

    algorithm_options = {'MADDPG': 'MADDPG'}

    # train_kwargs = {'avoid_collision': True,
    #                 'batch_size': 256,
    #                 'hidden_size':128,
    #                 'theta_opt': True,
    #                 'apply_err_sq': False,
    #                 'fair_service': True,
    #                 'map': map_otions['General4uavMap'],
    #                 'trans_scheme': transmission_scheme_options['Proposed'],
    #                 'algo': algorithm_options['MADDPG'],
    #                 'test_per_steps':30000}

    for ts in transmission_scheme_options:
        for fair_service in [True, False]:
            x = 'fair' if fair_service else 'unfair'
            file_name = f"maddpg_ts_{ts}_{x}"
            print(file_name)
            train_kwargs = {'avoid_collision': True,
                    'batch_size': 256, 
                    'hidden_size':128,
                    'theta_opt': True,
                    'apply_err_sq': False,
                    'fair_service': fair_service,
                    'map': map_otions['General4uavMap'],
                    'trans_scheme': ts,
                    'algo': algorithm_options['MADDPG'],
                    'test_per_steps':30000,
                    }
            train(args=args, train_kwards=train_kwargs, expname=file_name)

    # train(args=args, train_kwards=train_kwargs)
