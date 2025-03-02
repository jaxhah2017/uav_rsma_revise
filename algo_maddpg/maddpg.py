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

    def take_action(self, states, explore):
        # print(states)
        # print(states.shape)
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(self.n_agents)
        ]
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]



    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()

        all_target_act = []
        for pi, _next_obs in zip(self.target_policies, next_obs):
            with torch.no_grad():
                move_logits, power_logits, theta_logits = onehot_from_logits(pi(_next_obs))
                all_target_act.extend([
                    move_logits,
                    power_logits,
                    theta_logits
                ])

        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(-1, 1) + self.gamma * cur_agent.target_critic(target_critic_input) * (1 - done[i_agent].view(-1, 1))

        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out_move, cur_actor_out_power, cur_actor_out_theta = cur_agent.actor(obs[i_agent])
        cur_act_vf_in_move, cur_act_vf_in_power, cur_act_vf_in_theta = gumbel_softmax((cur_actor_out_move, cur_actor_out_power, cur_actor_out_theta))
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.extend([cur_act_vf_in_move, cur_act_vf_in_power, cur_act_vf_in_theta])
            else:
                move, power, theta = onehot_from_logits(pi(_obs))
                all_actor_acs.extend([move, power, theta])
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += ((cur_actor_out_move**2).mean() + (cur_actor_out_power**2).mean() + (cur_actor_out_theta**2).mean()) * 1e-3

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


