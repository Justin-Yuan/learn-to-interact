from collections import defaultdict
from copy import deepcopy 
import numpy as np 
import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete

from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from agents import DEFAULT_ACTION, DDPGAgent
from algorithms.maddpg.utils import switch_list


MSELoss = torch.nn.MSELoss()

class MADDPGEnsemble(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params=None, alg_types=None,
        gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
        #  discrete_action=False
        agent_map=None, **kwargs
    ):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.alg_types = alg_types
        self.agent_init_params = agent_init_params
        self.agent_pool = [
            DDPGAgent(lr=lr, hidden_dim=hidden_dim, **params)
            for params in agent_init_params    
        ]            
        # map from active agent to available agent pool
        self.agent_map = agent_map
        if agent_map is None:   # normal maddpg 
            self.nagents = len(alg_types)
        else:
            self.nagents = len(agent_map)
        # active agents, either for sampling or training 
        self.agent_indices = [-1] * self.nagents
        self.agents = [None] * self.nagents

        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        # self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0
        # summaries tracker 
        self.agent_losses = defaultdict(list)
       
    def get_summaries(self):
        return {"agent_losses": deepcopy(self.agent_losses)}

    def reset_summaries(self):
        self.agent_losses.clear()

    def sample_agents(self):
        """ sample agents to be used from agent_pool """
        sampled_idx = []
        for i in range(self.nagents):
            idx = int(np.random.choice(self.agent_map[i], 1)[0])
            sampled_idx.append(idx)
            self.agent_indices[i] = idx 
            self.agents[i] = self.agent_pool[idx]
        return sampled_idx

    # @property
    # def policies(self):
    #     return [a.policy for a in self.agents]

    # @property
    # def target_policies(self):
    #     return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def prep_training(self, device='cpu', batch_size=None):
        """ switch nn models to train mode """
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()

        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = a.policy.to(device)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = a.critic.to(device)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = a.target_policy.to(device)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = a.target_critic.to(device)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu', batch_size=None):
        """ switch nn models to eval mode """
        for a in self.agents:
            a.policy.eval()

        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = a.policy.to(device)
            self.pol_dev = device        

    ############################################ NOTE: save/init/restore
    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {
            'init_dict': self.init_dict,
            'agent_params': [a.get_params() for a in self.agent_pool] #self.agents]
        }
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(
        cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
        gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, 
        ensemble_size=1, ensemble_config=None,
        **kwargs
    ):
        """
        Instantiate instance of this class from multi-agent environment
        ensemble_config: "<agent_type>-<population size> <agent_type>-<population size> ..."
        """
        # prep agent pool info
        agent_list, agent_map = [], {}
        if ensemble_config is not None:
            # specifies pool size for each agent type 
            idx, temp = 0, {}
            for entry in ensemble_config.strip().split(" "):
                agent_type, agent_size = entry.split("-")
                agent_list += [agent_type] * agent_size
                # cache agent type to agent indices
                temp[agent_type] = list(range(idx, idx+agent_size))
                idx += agent_size 
            # create map from active agent idx to agent indices
            for i, agent_type in enumerate(env.agent_types):
                agent_map[i] = temp[agent_type]
        else:
            # specifies ensemble size for each active agent 
            for i, agent_type in enumerate(env.agent_types):
                agent_list += [agent_type] * ensemble_size
                agent_map[i] = list(range(i*ensemble_size, (i+1)*ensemble_size))
        
        obs_space_dict, act_space_dict = {}, {}
        for a_type in list(set(env.agent_types)):
            idx = agent_list.index(a_type)
            # record agent type to obs space
            obs_space_dict[a_type] = env.observation_space[idx]
            # record agent type to act space
            act_space_dict[a_type] = env.action_space[idx]


        # make init params for pool of agents
        agent_init_params = [] 
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in agent_list] #env.agent_types]
        obs_spaces = [obs_space_dict[a_type] for atype in agent_list]
        act_spaces = [act_space_dict[a_type] for atype in agent_list]

        for atype, algtype, acsp, obsp in zip(agent_list, alg_types, act_spaces, obs_spaces):           
            # used in initializing agent (including policy and critic)
            idx = agent_list.index(atype)
            agent_init_params.append({
                'algo_type': algtype,
                'act_space': acsp,
                'obs_space': obsp,
                'env_obs_space': switch_list(env.observation_space, idx),
                'env_act_space': switch_list(env.action_space, idx)
            })

        # make learaner 
        init_dict = {
            'gamma': gamma, 'tau': tau, 'lr': lr,
            'hidden_dim': hidden_dim,
            'alg_types': alg_types,
            'agent_init_params': agent_init_params,
            "agent_map": agent_map
            # 'discrete_action': discrete_action,
        }
        init_dict.update(kwargs)
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        # for a, params in zip(instance.agents, save_dict['agent_params']):
        for a, params in zip(instance.agent_pool, save_dict['agent_params']):
            a.load_params(params)
        return instance

    ############################################ NOTE: step/act
    def step(self, observations, explore=False):
        """ Take a step forward in environment with all agents
        Arguments:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Returnss:
            actions: List of action (np array or dict of it) for each agent
        """
        return [
            a.step(obs, explore=explore) 
            for a, obs in zip(self.agents, observations)
        ]

    ############################################ NOTE: update
    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def init_hidden(self, batch_size):
        """ for rnn policy, training or evaluation """
        for a in self.agents:
            a.init_hidden(batch_size) 

    def flatten_obs(self, obs, keys=None, ma=False):
        """ convert observations to single tensor, if dict, 
            concat by keys along last dimension 
        Arguments: 
            obs: (B,T,O) or dict of (B,T,O) or list []*N of it 
            keys: list of keys to concat (in order)
            ma: multi-agent flag, if true, obs is list 
        Returns:
            out: single tensor (B,T,sum_(O_k)) or lists []*N of it
        """
        def _flatten(x, keys=None):
            if not isinstance(x, dict):
                return x
            if keys is None:
                # maintain order for consistency 
                keys = sorted(list(x.keys()))
            return torch.cat([x[k] for k in keys], -1)
            
        if ma:
            return [_flatten(ob, keys) for ob in obs]
        else:
            return _flatten(obs, keys)

    def flatten_act(self, acs, keys=None, ma=False):
        """ convert actions to single tensor, if dict, 
            concat by keys along last dimension 
        Arguments:
            acs: dict of (B,T,A), or list []*N of it 
            keys: list of keys to concat (in order)
            ma: multi-agent flag, if true, obs is list 
        Returns:
            out: single tensor (B,T,sum_(A_k)) or list []*N of it 
        """
        def _flatten(x, keys=None):
            if not isinstance(x, dict):
                return x
            if DEFAULT_ACTION in x:
                # equivalent to single action     
                return x[DEFAULT_ACTION]
            if keys is None:
                keys = sorted(list(x.keys()))
            return torch.cat([x[k] for k in keys], -1)
            
        if ma:
            return [_flatten(ac, keys) for ac in acs]
        else:
            return _flatten(acs, keys)

    def add_virtual_dim(self, sample):
        """ since ddpg agent takes in obs (B,T,O)
            add a virtual T dim, so [(B,D)]*N  -> [(B,1,D)]*N or 
            [dict (B,D)]*N -> [dict (B,1,D)]*N
        """
        obs, acs, rews, next_obs, dones = sample # [(B,D)]*N
        obs = [
            {k: v.unsqueeze(1) for k, v in ob.items()}
            if isinstance(ob, dict) else ob.unsqueeze(1)
            for ob in obs
        ]
        next_obs = [
            {k: v.unsqueeze(1) for k, v in ob.items()}
            if isinstance(ob, dict) else ob.unsqueeze(1)
            for ob in next_obs
        ]
        acs = [
            {k: v.unsqueeze(1) for k, v in ac.items()}
            if isinstance(ac, dict) else ac.unsqueeze(1)
            for ac in acs
        ]
        rews = [rew.unsqueeze(1) for rew in rews]
        dones = [done.unsqueeze(1) for done in dones]
        return obs, acs, rews, next_obs, dones 

    def update(self, sample, agent_i, parallel=False, grad_norm=0.5):
        """ Update parameters of agent model based on sample from replay buffer
        Arguments:
            sample: [(B,D)]*N, obs, next_obs, action can be [dict (B,D)]*N
            NOTE: each's first element belong to agent_i
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
        """
        # [(B,1,D)]*N or [dict (B,1,D)]*N
        obs, acs, rews, next_obs, dones = self.add_virtual_dim(sample)   
        bs, ts, _ = obs[0].shape 
        # index into agent pool from active agent index
        curr_alg_type = self.alg_types[self.agent_indices[agent_i]]
        curr_agent = self.agents[agent_i]
        # since current agent is first in active agent list 
        curr_obs = obs[0]
        curr_ac = acs[0]
        curr_rew = rews[0]
        curr_nobs = next_obs[0]
        curr_done = dones[0]
        curr_agents = switch_list(self.agents, agent_i)

        # NOTE: Critic update
        curr_agent.critic_optimizer.zero_grad()

        # compute target actions
        with torch.no_grad():
            if curr_alg_type == 'MADDPG':
                all_trgt_acs = []   # [dict (B,1,A)]*N
                for i, nobs in enumerate(next_obs): # (B,1,O)
                    act_i = curr_agents[i].compute_action(nobs, target=True, requires_grad=False)
                    all_trgt_acs.append(act_i)
                # [(B,1,O)_i, ..., (B,1,A)_i, ...] -> (B,1,O*N+A*N)
                trgt_vf_in = torch.cat([
                    *self.flatten_obs(next_obs, ma=True), 
                    *self.flatten_act(all_trgt_acs, ma=True)
                ], dim=-1)
            else:  # DDPG
                act_i = curr_agent.compute_action(curr_nobs, target=True, requires_grad=False)
                # (B,1,O) + (B,1,A) -> (B,1,O+A)
                trgt_vf_in = torch.cat([
                    self.flatten_obs(curr_nobs), 
                    self.flatten_act(act_i)
                ], dim=-1)

            # bellman targets   # (B*T,1) -> (B*1,1) -> (B,1)
            target_q = curr_agent.compute_value(trgt_vf_in, target=True) 
            target_value = (curr_rew.view(-1, 1) + self.gamma * target_q *
                                (1 - curr_done.view(-1, 1)))   

        # Q func
        if curr_alg_type == 'MADDPG':
            vf_in = torch.cat([
                *self.flatten_obs(obs, ma=True), 
                *self.flatten_act(acs, ma=True)
            ], dim=-1)
        else:  # DDPG
            vf_in = torch.cat([
                self.flatten_obs(curr_obs),
                self.flatten_act(curr_ac)
            ], dim=-1)
        actual_value = curr_agent.compute_value(vf_in, target=False) # (B*T,1)

        # bellman errors
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        if grad_norm > 0:
            torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), grad_norm)
        curr_agent.critic_optimizer.step()

        # NOTE: Policy update
        curr_agent.policy_optimizer.zero_grad()

        # current agent action (deterministic, softened), dcit (B,T,A)
        curr_pol_out = curr_agent.compute_action(curr_obs, target=False, requires_grad=True) 

        if curr_alg_type == 'MADDPG':
            all_pol_acs = []
            for i, ob in zip(range(self.nagents), obs):
                if i == 0:    # insert current agent act to q input 
                    all_pol_acs.append(self.flatten_act(curr_pol_out))
                    # all_pol_acs.append(curr_pol_out)
                else: 
                    # p_act_i = curr_agents[i].compute_action(ob, target=False, requires_grad=False) 
                    p_act_i = self.flatten_act(acs[i])
                    all_pol_acs.append(p_act_i)
            # (B,T,O*N+A*N)s
            p_vf_in = torch.cat([
                *self.flatten_obs(obs, ma=True),
                *self.flatten_act(all_pol_acs, ma=True)
            ], dim=-1) 
        else:  # DDPG
            # (B,T,O+A)
            p_vf_in = torch.cat([
                self.flatten_obs(curr_obs),
                self.flatten_act(curr_pol_out)
            ], dim=-1) 
        
        # value function to update current policy
        p_value = curr_agent.compute_value(p_vf_in, target=False) # (B*T,1)
        pol_loss = -p_value.mean()

        # p regularization, scale down output (gaussian mean,std or logits)
        # reference: https://github.com/openai/maddpg/blob/master/maddpg/trainer/maddpg.py
        pol_reg_loss = torch.tensor(0.0)
        for k, v in curr_pol_out.items():
            pol_reg_loss += ((v.reshape(bs*ts, -1))**2).mean() * 1e-3

        pol_loss_total = pol_loss + pol_reg_loss
        pol_loss_total.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        if grad_norm > 0:
            torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), grad_norm)
        curr_agent.policy_optimizer.step()

        # NOTE: collect training statss 
        results = {}
        for k, v in zip(
            ["critic_loss", "policy_loss", "policy_reg_loss"], 
            [vf_loss, pol_loss, pol_reg_loss]
        ):
            key = "agent_{}/{}".format(agent_i, k)
            value = v.data.cpu().numpy()
            results[key] = value
            self.agent_losses[key].append(value)        
        return results





