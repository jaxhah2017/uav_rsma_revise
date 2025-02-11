from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW

from types import SimpleNamespace as SN

from algo_mha_drqn.utils import *

from algo_mha_drqn.buffer import ReplayBuffer

from algo_mha_drqn.agents import Agents

import random

# from algo_mha_drqn.agents import QMixer

import time

class MultiAgentQLearner:
    """Multi-Agent Q learning algorithm"""

    def __init__(self, env_info, args):
        self.args = args
        self.device = args.device
        # Extract drqn_env info
        self.gt_features_dim = env_info['gt_features_dim']
        self.other_features_dim = env_info['other_features_dim']
        self.n_heads = args.n_heads
        self.theta_opt = args.theta_opt
        self.state_shape = env_info['state_shape']
        self.n_moves = env_info['n_moves']
        self.n_powers = env_info['n_powers']
        self.n_agents = env_info['n_agents']
        self.n_thetas = env_info['n_thetas']

        self.policy_net = Agents(gt_features_dim=self.gt_features_dim,
                                 num_heads=self.n_heads,
                                 other_features_dim=self.other_features_dim,
                                 move_dim=self.n_moves,
                                 power_dim=self.n_powers,
                                 theta_dim=self.n_thetas,
                                 theta_opt=self.theta_opt,
                                 n_layers=args.n_layers,
                                 hidden_size=args.hidden_size).to(self.device)  # Policy Network
        self.target_net = Agents(gt_features_dim=self.gt_features_dim,
                                 num_heads=self.n_heads,
                                 other_features_dim=self.other_features_dim,
                                 move_dim=self.n_moves,
                                 power_dim=self.n_powers,
                                 theta_dim=self.n_thetas,
                                 theta_opt=self.theta_opt,
                                 n_layers=args.n_layers,
                                 hidden_size=args.hidden_size).to(self.device)  # Target Network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # print(f"policy network: \n{self.policy_net}")
        self.max_seq_len = args.max_seq_len if args.max_seq_len is not None else env_info['episode_limit']
        self.params = list(self.policy_net.parameters())  # Parameters to optimize
        self.mixer = None
        # QMix
        self.mixer = None
        if args.mixer:
            self.mixer = QMixer(self.state_shape, self.n_agents, args).to(self.device)  # QMixer
            self.target_mixer = deepcopy(self.mixer).to(self.device)
            print(f"mixer = \n{self.mixer}")
            self.params += list(self.mixer.parameters())

        self.gamma = args.gamma  # Discount factor
        self.polyak = args.polyak  # Interpolation factor in polyak averaging for target networks
        self.batch_size = args.batch_size  # Mini-batch size for SGD

        self.buffer = ReplayBuffer(args.replay_size, self.max_seq_len, self.theta_opt)  # Replay buffer
        self.loss_fn = nn.MSELoss()  # Loss function
        self.optimizer = AdamW(self.params, lr=args.lr)  # Optimizer
        self.anneal_lr = args.anneal_lr  # Whether lr annealing is used.
        if self.anneal_lr:
            lr_lambda = lambda epoch: max(0.4, 1 - epoch / 100)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda, verbose=True)
        self.double_q = args.double_q

    def init_hidden(self, batch_size=1):
        """Initializes RNN hidden states for all agents."""
        return self.policy_net.init_hidden().expand(self.n_agents * batch_size, -1)

    def take_actions(self, obs, h, eps_thres):
        """Selects actions following epsilon-greedy strategy"""
        obs_other_features = obs[0].to(self.device)
        obs_gt_features = obs[1].to(self.device)
        h = h.to(self.device)
        with torch.no_grad():
            start_time = time.time()
            if self.theta_opt:
                move_logits, power_logits, thetas_logits, h = self.policy_net(obs_gt_features, obs_other_features, h)
            else:
                move_logits, power_logits, h = self.policy_net(obs_gt_features, obs_other_features, h)
            end_time = time.time()
        if random.random() > eps_thres:
            moves = torch.argmax(move_logits, 1)
            powers = torch.argmax(power_logits, 1)
            if self.theta_opt:
                thetas = torch.argmax(thetas_logits, 1)
        else:
            moves = torch.randint(self.n_moves, size=(self.n_agents,), dtype=torch.long)
            powers = torch.randint(self.n_powers, size=(self.n_agents,), dtype=torch.long)
            if self.theta_opt:
                thetas = torch.randint(self.n_thetas, size=(self.n_agents,), dtype=torch.long)

        if self.theta_opt:
            acts = dict(moves=moves.tolist(), powers=powers.tolist(), thetas=thetas.tolist())
        else:
            acts = dict(moves=moves.tolist(), powers=powers.tolist())

        return acts, h, end_time - start_time

    def cache(self, obs, h, state, act, rew, next_obs, next_h, next_state, done, bad_mask):
        if self.args.share_reward:
            rew = rew.mean()
        obs_other_features = obs[0]
        obs_gt_features = obs[1]
        next_obs_other_features = next_obs[0]
        next_obs_gt_features = next_obs[1]

        # When done is True due to reaching episode limit, mute it.
        if self.theta_opt:
            transition = dict(obs_other_features=obs_other_features,
                            obs_gt_features=obs_gt_features,
                            h=h,
                            state=state,
                            act_moves=torch.tensor(act['moves'], dtype=torch.long).unsqueeze(1),
                            act_powers=torch.tensor(act['powers'], dtype=torch.long).unsqueeze(1),
                            act_thetas=torch.tensor(act['thetas'], dtype=torch.long).unsqueeze(1),
                            rew=torch.tensor(rew, dtype=torch.float32).reshape(1, -1),
                            next_obs_other_features=next_obs_other_features,
                            next_obs_gt_features=next_obs_gt_features,
                            next_h=(1 - done) * next_h, next_state=next_state,
                            done=torch.tensor((1 - bad_mask) * done, dtype=torch.float32).reshape(1, 1))
        else:
            transition = dict(obs_other_features=obs_other_features,
                            obs_gt_features=obs_gt_features,
                            h=h,
                            state=state,
                            act_moves=torch.tensor(act['moves'], dtype=torch.long).unsqueeze(1),
                            act_powers=torch.tensor(act['powers'], dtype=torch.long).unsqueeze(1),
                            rew=torch.tensor(rew, dtype=torch.float32).reshape(1, -1),
                            next_obs_other_features=next_obs_other_features,
                            next_obs_gt_features=next_obs_gt_features,
                            next_h=(1 - done) * next_h, next_state=next_state,
                            done=torch.tensor((1 - bad_mask) * done, dtype=torch.float32).reshape(1, 1))
        self.buffer.push(transition)

    def update(self):
        """Updates parameters of recurrent agents via BPTT."""

        assert len(self.buffer) >= self.batch_size, "Insufficient samples for update."

        samples = self.buffer.sample(self.batch_size)  # List of sequences
        batch = {k: [] for k in self.buffer.scheme}  # Dict holding batch of samples.

        # Construct input sequences.
        for t in range(self.max_seq_len):
            for k in batch.keys():
                x = [samples[i][k][t].to(self.device) for i in range(self.batch_size)]
                batch[k].append(cat(x))
        # Append next obs/h/state of the last timestep.
        for k in {'obs_other_features', 'obs_gt_features', 'h', 'state'}:
            x = [samples[i][k][self.max_seq_len].to(self.device) for i in range(self.batch_size)]
            batch[k].append(cat(x))

        # acts = torch.stack(batch['act']).to(self.device)
        act_moves = torch.stack(batch['act_moves']).to(self.device)
        act_powers = torch.stack(batch['act_powers']).to(self.device)
        if self.theta_opt:
            act_thetas = torch.stack(batch['act_thetas']).to(self.device)
        rews = torch.stack(batch['rew']).to(self.device)
        dones = torch.stack(batch['done']).to(self.device)
        h, h_targ = batch['h'][0].to(self.device), batch['h'][1].to(self.device)  # Get initial hidden states.

        agent_out_moves = []
        agent_out_powers = []
        if self.theta_opt:
            agent_out_thetas = []
        target_out_moves = []
        target_out_powers = []
        if self.theta_opt:
            target_out_thetas = []
        obs_other_features = [batch['obs_other_features'][t].to(self.device) for t in range(len(batch['obs_other_features']))]
        obs_gt_features = [batch['obs_gt_features'][t].to(self.device) for t in range(len(batch['obs_gt_features']))]


        for t in range(self.max_seq_len):
            # Policy network predicts the Q(s_{t},a_{t}) at current timestep.
            if self.theta_opt:
                move_logits, power_logits, theta_logits, h = self.policy_net(obs_gt_features[t], obs_other_features[t], h)
            else:
                move_logits, power_logits, h = self.policy_net(obs_gt_features[t],
                                                           obs_other_features[t],
                                                           h)
            agent_out_moves.append(move_logits)
            agent_out_powers.append(power_logits)
            if self.theta_opt:
                agent_out_thetas.append(theta_logits)
            # Target network predicts Q(s_{t+1}, a_{t+1}).
            with torch.no_grad():
                if self.theta_opt:
                    next_move_logits, next_power_logits, next_theta_logits, h_targ = self.target_net(obs_gt_features[t + 1],
                                                                                obs_other_features[t + 1],
                                                                                h_targ)
                else:
                    next_move_logits, next_power_logits, h_targ = self.target_net(obs_gt_features[t + 1],
                                                                              obs_other_features[t + 1],
                                                                              h_targ)
                # target_out.append([next_move_logits, next_power_logits])
                target_out_moves.append(next_move_logits)
                target_out_powers.append(next_power_logits)
                if self.theta_opt:
                    target_out_thetas.append(next_theta_logits)

        # Let policy network make predictions for next state of the last timestep in the sequence.
        if self.theta_opt:
            move_logits, power_logits, theta_logits, h = self.policy_net(obs_gt_features[self.max_seq_len],
                                                        obs_other_features[self.max_seq_len],
                                                        h)
        else:
            move_logits, power_logits, h = self.policy_net(obs_gt_features[self.max_seq_len],
                                                        obs_other_features[self.max_seq_len],
                                                        h)

        agent_out_moves.append(move_logits)
        agent_out_powers.append(power_logits)
        if self.theta_opt:
            agent_out_thetas.append(theta_logits)
        # Stack outputs of policy/target networks.
        if self.theta_opt:
            agent_out_moves, agent_out_powers, agent_out_thetas = torch.stack(agent_out_moves), torch.stack(agent_out_powers), torch.stack(agent_out_thetas)
            target_out_moves, target_out_powers, target_out_thetas = torch.stack(target_out_moves), torch.stack(target_out_powers), torch.stack(target_out_thetas)
        else:
            agent_out_moves, agent_out_powers = torch.stack(agent_out_moves), torch.stack(agent_out_powers)
            target_out_moves, target_out_powers = torch.stack(target_out_moves), torch.stack(target_out_powers)

        # Compute Q_{s_{t}, a_{t}} with policy network.
        q_moves_val = agent_out_moves[:-1].gather(2, act_moves)
        q_powers_val = agent_out_powers[:-1].gather(2, act_powers)
        if self.theta_opt:
            q_thetas_val = agent_out_thetas[:-1].gather(2, act_thetas)
        # Compute V_{s_{t+1}}.
        if not self.double_q:
            next_moves_val = target_out_moves.max(2, keepdim=True)[0]
            next_powers_val = target_out_powers.max(2, keepdim=True)[0]
            if self.theta_opt:
                next_thetas_val = target_out_thetas.max(2, keepdim=True)[0]
        else:
            next_moves = torch.argmax(agent_out_moves[1:].clone().detach(), dim=2, keepdim=True)
            next_powers = torch.argmax(agent_out_powers[1:].clone().detach(), dim=2, keepdim=True)
            if self.theta_opt:
                next_thetas = torch.argmax(agent_out_thetas[1:].clone().detach(), dim=2, keepdim=True)
            next_moves_val = target_out_moves.gather(2, next_moves)
            next_powers_val = target_out_powers.gather(2, next_powers)
            if self.theta_opt:
                next_thetas_val = target_out_thetas.gather(2, next_thetas)

        q_moves_val = q_moves_val.view(self.max_seq_len, self.batch_size, self.n_agents)
        q_powers_val = q_powers_val.view(self.max_seq_len, self.batch_size, self.n_agents)
        if self.theta_opt:
            q_thetas_val = q_thetas_val.view(self.max_seq_len, self.batch_size, self.n_agents)
        
        next_moves_val = next_moves_val.view(self.max_seq_len, self.batch_size, self.n_agents)
        next_powers_val = next_powers_val.view(self.max_seq_len, self.batch_size, self.n_agents)
        if self.theta_opt:
            next_thetas_val = next_thetas_val.view(self.max_seq_len, self.batch_size, self.n_agents)

        # Obtain target of update.
        rews, dones = rews.expand_as(next_moves_val), dones.expand_as(next_moves_val)
        target_moves_qvals = rews + self.gamma * (1 - dones) * next_moves_val
        rews, dones = rews.expand_as(next_powers_val), dones.expand_as(next_powers_val)
        target_powers_qvals = rews + self.gamma * (1 - dones) * next_powers_val
        if self.theta_opt:
            rews, dones = rews.expand_as(next_thetas_val), dones.expand_as(next_thetas_val)
            target_thetas_qvals = rews + self.gamma * (1 - dones) * next_thetas_val
        # Compute MSE loss.
        loss_moves = self.loss_fn(q_moves_val, target_moves_qvals)
        loss_powers = self.loss_fn(q_powers_val, target_powers_qvals)
        if self.theta_opt:
            loss_thetas = self.loss_fn(q_thetas_val, target_thetas_qvals)
            loss = (loss_moves + loss_powers + loss_thetas) / 3
        else:
            loss = (loss_moves + loss_powers) / 2

        # Call one step of gradient descent.
        self.optimizer.zero_grad()
        loss.backward()  # Back propagation
        nn.utils.clip_grad_value_(self.policy_net.parameters(), clip_value=1)  # Gradient-clipping
        self.optimizer.step()  # Call update.

        # Update the target network via polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.policy_net.parameters(), self.target_net.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

            if self.mixer is not None:
                for p, p_targ in zip(self.mixer.parameters(), self.target_mixer.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        if self.theta_opt:
            update_info = dict(LossQ=loss.item(),
                        move_QVals=q_moves_val.detach().cpu().numpy(),
                        power_QVals=q_powers_val.detach().cpu().numpy(),
                        theta_QVals=q_thetas_val.detach().cpu().numpy())
        else:
            update_info = dict(LossQ=loss.item(),
                    move_QVals=q_moves_val.detach().cpu().numpy(),
                    power_QVals=q_powers_val.detach().cpu().numpy(),)
                    # theta_QVals=q_thetas_val.detach().cpu().numpy())

        return update_info

    def save_model(self, path, stamp):
        checkpoint = dict()
        checkpoint.update(stamp)
        checkpoint['model_state_dict'] = self.policy_net.state_dict()
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(checkpoint, path)

        print(f"Save checkpoint to {path}.")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_net.eval()

        print(f"Load checkpoint from {path}.")


if __name__ == "__main__":
    env_info = dict(obs_shape=128, state_shape=128, n_actions=4, n_agents=3)
    args = SN()
    args.state_dim = 128
    args.action_dim = 5
    args.n_agents = 5
    args.gamma = 0.99
    args.polyak = 0.5
    args.batch_size = 32
    args.replay_size = 200
    args.lr = 0.01

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mulAgent = MultiAgentQLearner(env_info=env_info, args=args)

    obs = torch.randn(3, 128)
    h = mulAgent.init_hidden()
    eps_thres = 1
    acts, h = mulAgent.take_actions(obs, h, eps_thres)
    # print(acts, h)
    print(acts)
    print(h.shape)

    print(obs.shape)
