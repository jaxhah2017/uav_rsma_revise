import torch

import numpy as  np

import torch.nn.functional as F

import random

def onehot_from_logits(logits, eps=0.01):
    ''' 生成最优动作的独热（one-hot）形式 '''
    # argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # # 生成随机动作,转换成独热形式
    # rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
    #     np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    # ]], requires_grad=False).to(logits.device)
    # # 通过epsilon-贪婪算法来选择用哪个动作
    # return torch.stack([
    #     argmax_acs[i] if r > eps else rand_acs[i]
    #     for i, r in enumerate(torch.rand(logits.shape[0]))
    # ])

    move_logits = logits[0]
    power_logits = logits[1]
    theta_logits = logits[2]
    argmax_acs_move = (move_logits == move_logits.max(1, keepdim=True)[0]).float()
    argmax_acs_power = (power_logits == power_logits.max(1, keepdim=True)[0]).float()
    argmax_acs_theta = (theta_logits == theta_logits.max(1, keepdim=True)[0]).float()
    rand_acs_move = torch.autograd.Variable(torch.eye(move_logits.shape[1])[[
        np.random.choice(range(move_logits.shape[1]), size=move_logits.shape[0])
    ]], requires_grad=False).to(move_logits.device)
    rand_acs_power = torch.autograd.Variable(torch.eye(power_logits.shape[1])[[
        np.random.choice(range(power_logits.shape[1]), size=power_logits.shape[0])
    ]], requires_grad=False).to(power_logits.device)
    rand_acs_theta = torch.autograd.Variable(torch.eye(theta_logits.shape[1])[[
        np.random.choice(range(theta_logits.shape[1]), size=theta_logits.shape[0])
    ]], requires_grad=False).to(theta_logits.device)
    move = torch.stack([
        argmax_acs_move[i] if r > eps else rand_acs_move[i]
        for i, r in enumerate(torch.rand(move_logits.shape[0]))
    ])
    power = torch.stack([
        argmax_acs_power[i] if r > eps else rand_acs_power[i]
        for i, r in enumerate(torch.rand(power_logits.shape[0]))
    ])
    theta = torch.stack([
        argmax_acs_theta[i] if r > eps else rand_acs_theta[i]
        for i, r in enumerate(torch.rand(theta_logits.shape[0]))
    ])

    return move, power, theta
    


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    move_logits = logits[0]
    power_logits = logits[1]
    theta_logits = logits[2]
    y_move = gumbel_softmax_sample(move_logits, temperature)
    y_power = gumbel_softmax_sample(power_logits, temperature)
    y_theta = gumbel_softmax_sample(theta_logits, temperature)
    # y = gumbel_softmax_sample(logits, temperature)
    y_hard_move, y_hard_power, y_hard_theta = onehot_from_logits((y_move, y_power, y_theta), temperature)
    # y_hard_move = onehot_from_logits(y_move)
    # y_hard_power = onehot_from_logits(y_power)
    # y_hard_theta = onehot_from_logits(y_theta)
    # y_hard = onehot_from_logits(y)
    y_move = (y_hard_move.to(move_logits.device) - y_move).detach() + y_move
    y_power = (y_hard_power.to(power_logits.device) - y_power).detach() + y_power
    y_theta = (y_hard_theta.to(theta_logits.device) - y_theta).detach() + y_theta
    # y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y_move, y_power, y_theta

def check_args_sanity(args):
    """Checks sanity andQ avoids conflicts of arguments."""

    # Ensure specified cuda is used when it is available.
    if args.device == 'cuda' and torch.cuda.is_available():
        args.device = f'cuda:{args.cuda_index}'
    else:
        args.device = 'cpu'
    print(f"Choose to use {args.device}.")

    # When QMix is used, ensure a scalar reward is used.
    if hasattr(args, 'mixer'):
        if args.mixer and not args.share_reward:
            args.share_reward = True
            print("Since QMix is used, all agents are forced to share a scalar reward.")

    return args

def cat(data_list):
    """Concatenates list of inputs"""
    if isinstance(data_list[0], torch.Tensor):
        return torch.cat(data_list)
    # elif isinstance(data_list[0], dgl.DGLGraph):
    #     return dgl.batch(data_list)
    else:
        raise TypeError("Unrecognised observation type.")


def set_rand_seed(seed=3407):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

import pickle
def save_var(var_path, var):
    var_n = var_path + '.pickle'
    with open(var_n, 'wb') as f:
        pickle.dump(var, f)
def load_var(var_path):
    # 从文件中读取变量
    var_n = var_path + '.pickle'
    with open(var_n, 'rb') as f:
        my_var = pickle.load(f)

    return my_var

import json
def convert_json(obj):
    def is_json_serializable(v):
        try:
            json.dumps(v)
            return True
        except:
            return False

    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v)
                    for k,v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v)
                        for k,v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)

import os
def save_config(output_dir, config):
    config_json = convert_json(config)
    output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
    print(output)
    with open(os.path.join(output_dir, "config.json"), 'w') as out:
        out.write(output)

import glob
def del_file(file_suffix):
    file_list = glob.glob('*.' + file_suffix)
    for file_path in file_list:
        os.remove(file_path)