from agents import Agents_actor, Agents_critic

import torch

import argparse

from utils import *

class DDPG:
    """algo DDPG"""
    def __init__(self, env_info, args):
        self.gt_features_dim = env_info['gt_features_dim']
        self.other_features_dim = env_info['other_features_dim']
        self.state_shape = env_info['state_shape']
        self.n_moves = env_info['n_moves']
        self.n_powers = env_info['n_powers']
        self.n_agents = env_info['n_agents']
        self.n_thetas = env_info['n_thetas']
        self.args = args

        self.device = args.device

        self.actor_in_dim = np.prod(self.gt_features_dim) + np.prod(self.other_features_dim)
        self.critic_in_dim = self.n_agents * (np.prod(self.gt_features_dim) + np.prod(self.other_features_dim) + self.n_moves + self.n_powers + self.n_thetas)

        self.actor = Agents_actor(input_dim=self.actor_in_dim,
                                 move_dim=self.n_moves,
                                 power_dim=self.n_powers,
                                 theta_dim=self.n_thetas,
                                 n_layers=args.n_layers,
                                 hidden_size=args.hidden_size).to(self.device)  
        self.target_actor = Agents_actor(input_dim=self.actor_in_dim,
                                 move_dim=self.n_moves,
                                 power_dim=self.n_powers,
                                 theta_dim=self.n_thetas,
                                 n_layers=args.n_layers,
                                 hidden_size=args.hidden_size).to(self.device)  
        self.critic = Agents_critic(input_dim=self.critic_in_dim,
                                 n_layers=args.n_layers,
                                 hidden_size=args.hidden_size).to(self.device)  
        self.target_critic = Agents_critic(input_dim=self.critic_in_dim,
                                 n_layers=args.n_layers,
                                 hidden_size=args.hidden_size).to(self.device)  
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.anneal_lr = args.anneal_lr  # Whether lr annealing is used.
        if self.anneal_lr:
            lr_lambda = lambda epoch: max(0.4, 1 - epoch / 100)
            self.actor_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=lr_lambda, verbose=True)
            self.critic_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=lr_lambda, verbose=True)

    def take_action(self, state, explore=False):
        action = self.actor(state)
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)

        action_move = action[0]
        action_power = action[1]
        action_theta = action[2]

        move = action_move.detach().cpu().numpy()[0]
        power = action_power.detach().cpu().numpy()[0]
        theta = action_theta.detach().cpu().numpy()[0]

        actions = dict(moves=move, powers=power, thetas=theta)

        return actions

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)



if __name__  == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--anneal_lr', type=bool, default=True, help='anneal learning rate')
    parser.add_argument('--n_layers', type=int, default=2, help='n_layers')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden_size')

    
    args = parser.parse_args()

    print(args)
    
    ddpg = DDPG(env_info = {}, args=args)

    
    print(ddpg.lr_scheduler)
        
        
        

