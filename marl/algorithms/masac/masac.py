from collections import defaultdict
from copy import deepcopy 
import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete

from agents import DEFAULT_ACTION, SACAgent
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax


MSELoss = torch.nn.MSELoss()

class MASAC(object):
    """
    Wrapper class for SAC-esque (i.e. also MASAC) agents in multi-agent task
    reference: https://arxiv.org/pdf/1812.05905.pdf
    reference2: https://github.com/astooke/rlpyt/blob/master/rlpyt/algos/qpg/sac.py
                https://github.com/astooke/rlpyt/blob/master/rlpyt/agents/qpg/sac_agent.py
    """
    def __init__(self, agent_init_params=None, alg_types=None,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                #  discrete_action=False
                target_entropy=10.0, **kwargs
        ):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (SAC
                                       or MASAC)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [SACAgent(lr=lr, hidden_dim=hidden_dim, **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr

        # sac specific params
        self.target_entropy = target_entropy

        # self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0
        # summaries tracker 
        self.agent_losses = defaultdict(list)

    def get_summaries(self):
        return {"agent_losses": deepcopy(self.agent_losses)}

    def reset_summaries(self):
        self.agent_losses.clear()

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    def prep_training(self, device='cpu'):
        """ switch nn models to train mode """
        for a in self.agents:
            a.policy.train()
            a.critic1.train()
            a.target_critic1.train()
            a.critic2.train()
            a.target_critic2.train()
            
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = a.policy.to(device)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic1 = a.critic1.to(device)
                a.critic2 = a.critic2.to(device)
            self.critic_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic1 = a.target_critic1.to(device)
                a.target_critic2 = a.target_critic2.to(device)
            self.trgt_critic_dev = device
     
    def prep_rollouts(self, device='cpu'):
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
        """ Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MASAC", adversary_alg="MASAC",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, 
                      rnn_policy=False, rnn_critic=False, **kwargs
    ):
        """ Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]

        for algtype, acsp, obsp in zip(alg_types, env.action_space, env.observation_space):           
            # used in initializing agent (including policy and critic)
            agent_init_params.append({
                'algo_type': algtype,
                'act_space': acsp,
                'obs_space': obsp,
                'rnn_policy': rnn_policy,
                'rnn_critic': rnn_critic,
                'env_obs_space': env.observation_space,
                'env_act_space': env.action_space
            })
        # also put agent init dicts to masac init_dict
        init_dict = {
            'gamma': gamma, 'tau': tau, 'lr': lr,
            'hidden_dim': hidden_dim,
            'alg_types': alg_types,
            'agent_init_params': agent_init_params,
            # 'discrete_action': discrete_action,
        }
        # sac specific configs
        init_dict.update(kwargs)

        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """ Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
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
        actions, info = [], {}
        log_probs = []

        # collect & evaluate actions
        for a, obs in zip(self.agents, observations):
            act_d, log_prob_d = a.step(obs, explore=explore) 
            actions.append(act_d)   # dict (B,A)
            log_probs.append(log_prob_d)    # dict (B,1)
        info["log_probs"] = log_probs  # [dict (B,1)]*N

        return actions, info
       
    ############################################ NOTE: update
    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic1, a.critic1, self.tau)
            soft_update(a.target_critic2, a.policy2, self.tau)
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
            acs: (B,T,A), ordict of (B,T,A), or list []*N of it 
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

    def contract_logprob(self, log_probs, keys=None, ma=False):
        """ convert log probs to joint log prob
        Arguments:
            log_probs: (B,T,1), or dict of (B,T,1), or list []*N of it 
            keys: list of keys to concat (in order)
            ma: multi-agent flag, if true, obs is list 
        Returns:
            out: single tensor (B,T,1) or list []*N of it 
        """
        def _contract(x, keys=None):
            if not isinstance(x, dict):
                return x 
            if DEFAULT_ACTION in x:
                # equivalent to single action     
                return x[DEFAULT_ACTION]
            if keys is None:
                keys = sorted(list(x.keys()))
            return torch.sum(
                torch.cat([x[k] for k in keys], -1), -1
            )   # [(B,T,1)]*K -> (B,T,K) -> (B,T,1)
            
        if ma:
            return [_contract(lp, keys) for lp in log_probs]
        else:
            return _contract(log_probs, keys)

    def update(self, sample, agent_i, parallel=False, grad_norm=0.5):
        """ Update parameters of agent model based on sample from replay buffer
        Arguments:
            sample: [(B,T,D)]*N, obs, next_obs, action, logprobs can be [dict (B,T,D)]*N
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
        """
        obs, acs, rews, next_obs, dones, logprobs = sample # each [(B,T,D)]*N 
        bs, ts, _ = obs[agent_i].shape 
        self.init_hidden(bs)  # use pre-defined init hiddens 
        curr_agent = self.agents[agent_i]

        # entropy temperature param
        alpha = curr_agent.log_alpha.get_alpha().detach()

        # NOTE: Critic update
        curr_agent.critic_optimizer.zero_grad()

        # compute target actions
        if self.alg_types[agent_i] == 'MASAC':
            all_trgt_acs = []   # [dict (B,T,A)]*N
            all_trgt_logprobs = []  # [dict (B,T,1)]*N

            for i, nobs in enumerate(next_obs): # (B,T,O)
                with torch.no_grad():
                    act_i, log_prob_i, _ = self.agents[i].compute_action_logprob(nobs)
                all_trgt_acs.append(act_i)
                all_trgt_logprobs.append(log_prob_i)

            # [(B,T,O)_i, ..., (B,T,A)_i, ...] -> (B,T,O*N+A*N)
            trgt_vf_in = torch.cat([
                *self.flatten_obs(next_obs, ma=True), 
                *self.flatten_act(all_trgt_acs, ma=True)
            ], dim=-1)

            # log prob of target action, [(B,T,1)]*N
            target_a_logprob = self.contract_logprob(all_trgt_logprobs, ma=True)
            # [(B,T,1)]*N -> (B,T,N) -> (B,T,1)
            target_a_logprob = torch.sum(
                torch.cat(target_a_logprob, -1), -1
            )

        else:  # SAC
            with torch.no_grad():
                act_i, log_prob_i, _ = curr_agent.compute_action_logprob(next_obs[agent_i])
            # (B,T,O) + (B,T,A) -> (B,T,O+A)
            trgt_vf_in = torch.cat([
                self.flatten_obs(next_obs[agent_i]), 
                self.flatten_act(act_i)
            ], dim=-1)

            # log prob of target action, (B,T,1)
            target_a_logprob = self.contract_logprob(log_prob_i)

        # bellman targets
        target_q1, target_q2 = curr_agent.compute_value(trgt_vf_in, target=True) # (B*T,1)
        target_a_logprob = target_a_logprob.reshape(-1,1).detach()   # (B*T,1)

        target_q = torch.min(target_q1, target_q2) - alpha * target_a_logprob
        target_value = (rews[agent_i].view(-1, 1) + self.gamma * target_q *
                            (1.0 - dones[agent_i].view(-1, 1)))   # (B*T,1)

        # Q func
        if self.alg_types[agent_i] == 'MASAC':
            vf_in = torch.cat([
                *self.flatten_obs(obs, ma=True), 
                *self.flatten_act(acs, ma=True)
            ], dim=-1)
        else:  # DDPG
            vf_in = torch.cat([
                self.flatten_obs(obs[agent_i]),
                self.flatten_act(acs[agent_i])
            ], dim=-1)
        q1, q2 = curr_agent.compute_value(vf_in, target=False) # (B*T,1)

        # bellman errors
        vf_loss1 = MSELoss(q1, target_value.detach())
        vf_loss1.backward()
        vf_loss2 = MSELoss(q2, target_value.detach())
        vf_loss2.backward()
        if parallel:
            average_gradients(curr_agent.critic1)
            average_gradients(curr_agent.critic2)
        if grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(curr_agent.critic1.parameters(), grad_norm)
            torch.nn.utils.clip_grad_norm_(curr_agent.critic2.parameters(), grad_norm)
        curr_agent.critic1_optimizer.step()
        curr_agent.critic2_optimizer.step()

        # NOTE: Policy update
        curr_agent.policy_optimizer.zero_grad()

        # current agent action (deterministic, softened), dcit (B,T,A)
        curr_pol_out, curr_log_prob, _ = curr_agent.compute_action_logprob(obs[agent_i]) 
        a_log_prob = self.contract_logprob(log_prob_d)

        if self.alg_types[agent_i] == 'MASAC':
            all_pol_acs = []
            all_pol_logprobs = []

            for i, pi, ob in zip(range(self.nagents), self.policies, obs):
                if i == agent_i:    # insert current agent act to q input 
                    all_pol_acs.append(self.flatten_act(curr_pol_out))
                    # current agent log prob (backprop-able)
                    all_pol_logprobs.append(curr_log_prob)
                else: 
                    # TODO: need other agents' log probs as well
                    # p_act_i = self.agents[i].compute_action(ob, target=False, requires_grad=False) 
                    p_act_i = self.flatten_act(acs[i])
                    all_pol_acs.append(p_act_i)
                    # other agents' log probs (during sampling)
                    all_pol_logprobs.append(logprobs[i])

            # (B,T,O*N+A*N)s
            p_vf_in = torch.cat([
                *self.flatten_obs(obs, ma=True),
                *self.flatten_act(all_pol_acs, ma=True)
            ], dim=-1) 

            # [dict (B,T,1)]*N -> [(B,T,1)]*N -> (B,T,1) 
            a_log_prob = self.contract_logprob(all_pol_logprobs, ma=True)
            a_log_prob = torch.sum(
                torch.cat(a_log_prob, -1), -1
            )

        else:  # DDPG
            # (B,T,O+A)
            p_vf_in = torch.cat([
                self.flatten_obs(obs[agent_i]),
                self.flatten_act(curr_pol_out)
            ], dim=-1) 

            # dict (B,T,1) -> (B,T,1)
            a_log_prob = self.contract_logprob(curr_log_prob)
        
        # KL loss between alpha log prob & target policy value function 
        p_value1, p_value2 = curr_agent.compute_value(p_vf_in, target=False) # (B*T,1)
        p_value_target = torch.min(p_value1, p_value2)
        pol_loss = alpha * a_log_prob - p_value_target
        
        # NOTE: this is optional (not in SAC)
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
            torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), grad_norm)
        curr_agent.policy_optimizer.step()

        # NOTE: Alpha (entropy) update
        alpha_loss = -curr_agent.log_alpha() * (a_log_prob.detach() + self.target_entropy)
        alpha_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        if grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), grad_norm)
        curr_agent.alpha_optimizer.step()
        
        # NOTE: collect training statss 
        results = {}
        for k, v in zip(
            ["critic1_loss", "critic2_loss", "policy_loss", "policy_reg_loss", "alpha_loss"], 
            [vf_loss1, vf_loss2, pol_loss, pol_reg_loss, alpha_loss]
        ):
            key = "agent_{}/{}".format(agent_i, k)
            value = v.data.cpu().numpy()
            results[key] = value
            self.agent_losses[key].append(value)        
        return results

    

    