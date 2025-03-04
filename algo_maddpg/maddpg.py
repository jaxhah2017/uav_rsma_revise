from ddpg import DDPG

import torch

from utils import *

class MADDPG:
    def __init__(self, env_info, args):
        self.agents = []
        self.n_agents = env_info['n_agents']

        for i in range(self.n_agents):
            self.agents.append(
                DDPG(env_info, args)
            )
        
        self.gamma = args.gamma
        self.tau = args.tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = args.device
        self.anneal_lr = args.anneal_lr

    def lr_step(self):
        for agent in self.agents:
            agent.actor_lr_scheduler.step()
            agent.actor_lr_scheduler.step()

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, hs, explore):
        # print(states)
        # print(states.shape)
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(self.n_agents)
        ]
        hs = [
            hs[i].to(self.device) for i in range(self.n_agents)
        ]

        actions = []    
        h2s = []
        for agent, state, h in zip(self.agents, states, hs):
            action, h2 = agent.take_action(state, h, explore)
            actions.append(action)
            h2s.append(h2)

        return actions, h2s

    def init_hidden(self):
        h = [agent.init_hidden() for agent in self.agents]
        
        return h

    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done, h, next_h = sample
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()

        all_target_act = []
        for pi, _next_obs, _next_h in zip(self.target_policies, next_obs, next_h):
            with torch.no_grad():
                move, power, theta, _ = pi(_next_obs, _next_h)
                actions = (move, power, theta)
                # actions, _ = pi(_next_obs, _next_h)
                move_logits, power_logits, theta_logits = onehot_from_logits(actions)
#                 move_logits, power_logits, theta_logits, _ = onehot_from_logits(pi(_next_obs))
                all_target_act.extend([
                    move_logits,
                    power_logits,
                    theta_logits
                ])

        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_q, _ = cur_agent.target_critic(target_critic_input, next_h[i_agent])
        target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * target_q * (1 - done[i_agent].view(-1, 1))
        # target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * cur_agent.target_critic(target_critic_input) * (1 - done[i_agent].view(-1, 1))

        critic_input = torch.cat((*obs, *act), dim=1)
        q_value, _ = cur_agent.critic(critic_input, h[i_agent])
        # critic_value = cur_agent.critic(critic_input) 
        critic_loss = self.critic_criterion(q_value, target_critic_value.detach())
        # critic_loss = self.critic_criterion(critic_value,
        #                                     target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out_move, cur_actor_out_power, cur_actor_out_theta, _ = cur_agent.actor(obs[i_agent], h[i_agent])
        cur_actor_out = (cur_actor_out_move, cur_actor_out_power, cur_actor_out_theta)
        # cur_actor_out_move, cur_actor_out_power, cur_actor_out_theta = cur_agent.actor(obs[i_agent])
        cur_act_vf_in_move, cur_act_vf_in_power, cur_act_vf_in_theta = gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        for i, (pi, _obs, _h) in enumerate(zip(self.policies, obs, h)):
            if i == i_agent:
                all_actor_acs.extend([cur_act_vf_in_move, cur_act_vf_in_power, cur_act_vf_in_theta])
            else:
                with torch.no_grad():
                    move, power, theta, _ = pi(_obs, _h)
                    actions = (move, power, theta)
                    # move, power, theta = onehot_from_logits(pi(_obs))
                    move, power, theta = onehot_from_logits(actions)
                    all_actor_acs.extend([move, power, theta])
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        q_value, _ = cur_agent.critic(vf_in, h[i_agent])
        actor_loss = -q_value.mean()
        # actor_loss = -cur_agent.critic(vf_in, h[i_agent]).mean()
        actor_loss += ((cur_actor_out_move**2).mean() + (cur_actor_out_power**2).mean() + (cur_actor_out_theta**2).mean()) * 1e-3
        # actor_loss += (cur_actor_out.pow(2).mean()) * 1e-3

        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)

    def save_model(self, path, stamp={}):
        checkpoint = dict()
        checkpoint.update(stamp)
        for i, agent in enumerate(self.agents):
            actor_name = f"agent{i}_actor"
            critic_name = f"agent{i}_critic"
            checkpoint[actor_name] = agent.actor.state_dict()
            checkpoint[critic_name] = agent.critic.state_dict()
        torch.save(checkpoint, path)

        print(f"Save checkpoint to {path}.")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        for i in range(self.n_agents):
            actor_name = f"agent{i}_actor"
            critic_name = f"agent{i}_critic"
            self.agents[i].actor.load_state_dict(checkpoint[actor_name])
            self.agents[i].actor.eval()
            self.agents[i].critic.load_state_dict(checkpoint[critic_name])


from types import SimpleNamespace as SN

if __name__ == "__main__":
    env_info = dict(obs_shape=128,
                    state_shape=128, 
                    n_actions=4, 
                    n_agents=4, 
                    gt_features_dim=10,
                    other_features_dim=6,
                    n_moves=5,
                    n_powers=5,
                    n_thetas=5)
    args = SN()

    args.gamma = 0.99
    args.tau = 0.1
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.anneal_lr = False
    args.n_layers = 2
    args.hidden_size = 256
    args.lr = 0.01
    
    learner = MADDPG(env_info=env_info, args=args)
    
    learner.save_model('check_point')

    learner.load_model('check_point')


