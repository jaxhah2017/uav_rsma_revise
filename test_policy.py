from types import SimpleNamespace as SN

from algo_mha_drqn.utils import *

from multi_uav_env.multi_ubs_env import MultiUbsRsmaEvn

from algo_mha_drqn.ma_learner import MultiAgentQLearner

from multi_uav_env.maps import *

from draw_tools import *

import shutil

def load_and_run_policy(model_path='', config_path=''):
    import json
    with open(config_path) as user_file:
        file_contents = user_file.read()

    parsed_json = json.loads(file_contents)
    config = list(parsed_json.values())[0]
    args = SN(**config)
    args.map = General4uavMap
    # print(args)
    # Set random seeds.
    set_rand_seed(args.seed)
    if args.cuda_deterministic:
        # Sacrifice performance to ensure reproducibility.
        # Refer to https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    env = MultiUbsRsmaEvn(args)
    env_info = env.get_env_info()
    learner = MultiAgentQLearner(env_info, args)
    learner.load_model(model_path)
    t = 0
    Reward = []
    with torch.no_grad():
        d = False
        (o, s, init_info), h = env.reset(), learner.init_hidden()
        while not d:
            a, h2, inference_time = learner.take_actions(o, h, 0.05)
            o2, s2, r, d, info = env.step(a)
            Reward.append(r)
            o, s, h = o2, s2, h2
        if d:
            useful_data = env.get_data()
            uav_traj = useful_data['traj']
            jamming_powers = useful_data['jamming']
            fair_index = useful_data['fair_idx']
            secrecy_rate = useful_data['secrecy_rate']
            throughput = useful_data['throughput']
            n_uav = str(env_info['n_ubs'])

            data = dict(expname=n_uav + "uav",
                        init_info=init_info,
                        uav_traj=uav_traj,
                        jamming_powers=jamming_powers,
                        fair_index=fair_index,
                        secrecy_rate=secrecy_rate,
                        throughput=throughput,
                        reward=Reward,
                        args=args
                        )
            print("智能体与环境交互第{}次, "
                      "ep_ret = {}, "
                      "ssr_system_rate = {}, "
                      "total_throughput = {}".format(1, info['EpRet'], info['Ssr_Sys'], info['total_throughput']))
    return data, args


if __name__ == '__main__':
    exp_name = 'exp1/'
    exp_path = './mha_drqn_data/' + exp_name
    config_path = exp_path + 'config.json'
    test_ret_path = exp_path + 'vars/test_p_ret'
    train_ret_path = exp_path + 'vars/ep_ret_list'
    model_path = exp_path + 'checkpoints/checkpoint_epoch'

    data_save_path = exp_path + '/vars/data'
    model_save_path = exp_path + '/vars/best_cpk.pt'
    start = 200

    pic_save_path = exp_path + 'pics'
    if not os.path.exists(pic_save_path):
        os.mkdir(pic_save_path)    

    test_returns = load_var(test_ret_path)
    plot_test_returns(test_returns=test_returns, 
                      pic_save_path=pic_save_path,
                      smooth=False,
                      smooth_beta=0.9)

    saveif = True
    overview = False
    if overview:
        for _ in range(10000):
            model_path2 = model_path + str(start) + '.pt'

            data, args = load_and_run_policy(model_path=model_path2, config_path=config_path)

            n_uav = len(data['init_info']['uav_init_pos'])

            if n_uav == 4:
                plot4uav_traj(init_info=data['init_info'], uav_traj=data['uav_traj'], jamming_power=data['jamming_powers'], args=args, pic_path=pic_path)
            elif n_uav == 2:
                plot2uav_traj(init_info=data['init_info'], uav_traj=data['uav_traj'], jamming_power=data['jamming_powers'], args=args, pic_path=pic_path)            

            if saveif:
                shutil.copyfile(model_path2, model_save_path)
                save_var(data_save_path, data)
                break

            start += 10

    draw = True
    if draw and not overview:
        data = load_var(data_save_path)
        print(data.keys())
        print(data['throughput'])
        plot_fair_index(epi_fair_index=data['fair_index'], pic_save_path=pic_save_path)
        plot_secrecy_rate(epi_secrecy_rate=data['secrecy_rate'], pic_save_path=pic_save_path)
        plot_throughput(epi_throughput=data['throughput'], pic_save_path=pic_save_path)