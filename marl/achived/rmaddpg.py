import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent

MSELoss = torch.nn.MSELoss()

class RMADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params=None, alg_types=None,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
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
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim, **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

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

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [
            a.step(obs, explore=explore) 
            for a, obs in zip(self.agents, observations)
        ]

    def parse_sample(self, sample, fields=None):
        """ extract training samples to specific format 
        Arguments:
            sample: a SampleBatch or EpisodeBatch 
            fields: list of keys to retrieve for 
        Returns:
            obs, acs, rews, next_obs, dones: each is [(B,T,D)]*N 
        """
        if fields is None:
            fields = ["obs", "action", "reward", "next_obs", "done"]    # default 
        parsed = [[] for _ in range(self.nagents)]
        for f_i, f in enumerate(fields):
            for a_i in range(self.nagents):
                key = f + "_{}".format(a_i)
                parsed[f_i].append(sample[key]) # (B,T,D)
        return parsed 

    def compute_q_val(self, agent_i, critic, vf_in, bs=None, ts=None):
        """ training critic forward with specified policy 
        Arguments:
            agent_i: index to agent; critic: critic network to agent; vf_in: (B,T,K);
            bs: batch size; ts: length of episode
        Returns:
            q: (B*T,1)
        """
        agent = self.agents[agent_i]
        if agent.rnn_critic:
            q = []   # (B,1)*T
            h_t = agent.critic_hidden_states.clone() # (B,H)
            for t in range(ts):
                q_t, h_t = critic(vf_in[:,t], h_t)
                q.append(q_t)
            q = torch.stack(q, 0).permute(1,0,2)   # (T,B,1) -> (B,T,1)
            q = q.reshape(bs*ts, -1)  # (B*T,1)
        else:
            # (B,T,D) -> (B*T,1)
            q = critic(vf_in.reshape(bs*ts, -1))
        return q 

    def compute_action(self, agent_i, pi, obs, bs=None, ts=None):
        """ traininsg actor forward with specified policy 
        Arguments:
            agent_i: index to agent; pi: policy to agent; obs: (B,T,O);
            bs: batch size; ts: length of episode
        Returns:
            act_i: (B,T,A)
        """
        agent = self.agents[agent_i]
        if agent.rnn_policy:
            act_i = []  # [(B,A)]*T
            h_i_t = agent.policy_hidden_states.clone() # (B,H)
            for t in range(ts):
                act_i_t, h_i_t = pi(obs[:,t], h_i_t)
                act_i.append(act_i_t)
            act_i = torch.stack(act_i, 0).permute(1,0,2)   # (B,T,A)
        else:
            stacked_obs = obs.reshape(bs*ts, -1)  # (B*T,O)
            act_i = pi(stacked_obs).reshape(bs, ts, -1)    # (B*T,A) -> (B,T,A)  
        return act_i 
        

    ############################################ NOTE: update
    def update(self, sample, agent_i, parallel=False, grad_norm=0.5):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: EpisodeBatch, use sample[key_i] to get a specific 
                    array of obs, action, etc for agent i
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
        """
        obs, acs, rews, next_obs, dones = self.parse_sample(sample) # [(B,T,D)]*N  
        bs, ts, _ = obs[0].shape
        self.init_hidden(bs)  # use pre-defined init hiddens 
        curr_agent = self.agents[agent_i]

        # NOTE: critic update
        curr_agent.critic_optimizer.zero_grad()

        # compute target actions
        if self.alg_types[agent_i] == 'MADDPG':
            all_trgt_acs = []   # [(B,T,A)]*N
            for i, (pi, nobs) in enumerate(zip(self.target_policies, next_obs)):
                # nobs: (B,T,O)
                act_i = self.compute_action(i, pi, nobs, bs=bs, ts=ts)
                all_trgt_acs.append(act_i)  # [(B,T,A)]
            
            if self.discrete_action:    # one-hot encode action
                all_trgt_acs = [onehot_from_logits(
                    act_i.reshape(bs*ts,-1)
                ).reshape(bs,ts,-1) for act_i in all_trgt_acs] 

            # critic input, [(B,T,O)_i, ..., (B,T,A)_i, ...] -> (B,T,O*N+A*N)
            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=-1)
        else:  # DDPG
            act_i = self.compute_action(agent_i, curr_agent.target_policy, 
                        next_obs[agent_i], bs=bs, ts=ts)

            if self.discrete_action:
                act_i = onehot_from_logits(
                    act_i.reshape(bs*ts, -1)
                ).reshape(bs, ts, -1) 

            # (B,T,O) + (B,T,A) -> (B,T,O+A)
            trgt_vf_in = torch.cat((next_obs[agent_i], act_i), dim=-1)

        # bellman targets
        target_q = self.compute_q_val(agent_i, curr_agent.target_critic, 
                        trgt_vf_in, bs=bs, ts=ts)   # (B*T,1)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma * target_q *
                            (1 - dones[agent_i].view(-1, 1)))   # (B*T,1)

        # Q func
        if self.alg_types[agent_i] == 'MADDPG':
            vf_in = torch.cat((*obs, *acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
        actual_value = self.compute_q_val(agent_i, curr_agent.critic, 
                            vf_in, bs=bs, ts=ts)    # (B*T,1)

        # bellman errors
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        if grad_norm > 0:
            torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), grad_norm)
        curr_agent.critic_optimizer.step()

        # NOTE: policy update
        curr_agent.policy_optimizer.zero_grad()

        # current agent action (deterministic, softened)
        curr_pol_out = self.compute_action(i, curr_agent.policy, 
                                obs[agent_i], bs=bs, ts=ts) # (B,T,A)
        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_vf_in = gumbel_softmax(
                                curr_pol_out.reshape(bs*ts, -1), hard=True
                            ).reshape(bs, ts, -1) 
        else:
            curr_pol_vf_in = curr_pol_out

        if self.alg_types[agent_i] == 'MADDPG':
            all_pol_acs = []
            for i, pi, ob in zip(range(self.nagents), self.policies, obs):
                if i == agent_i:
                    # insert current agent act to q input 
                    all_pol_acs.append(curr_pol_vf_in)
                else: 
                    p_act_i = self.compute_action(i, pi, ob, bs=bs, ts=ts) # (B,T,A)
                    if self.discrete_action:
                        p_act_i = onehot_from_logits(
                                    p_act_i.reshape(bs*ts, -1)
                                ).reshape(bs, ts, -1) 
                    all_pol_acs.append(p_act_i)
            p_vf_in = torch.cat((*obs, *all_pol_acs), dim=-1) # (B,T,O*N+A*N)
        else:  # DDPG
            p_vf_in = torch.cat((obs[agent_i], curr_pol_vf_in), dim=-1) # (B,T,O+A)
        
        # value function to update current policy
        p_value = self.compute_q_val(agent_i, curr_agent.critic, 
                        p_vf_in, bs=bs, ts=ts)   # (B*T,1)
        pol_loss = -p_value.mean()
        # p regularization, scale down output (gaussian mean,std or logits)
        # reference: https://github.com/openai/maddpg/blob/master/maddpg/trainer/maddpg.py
        pol_loss += ((curr_pol_out.reshape(bs*ts, -1))**2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        if grad_norm > 0:
            torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), grad_norm)
        curr_agent.policy_optimizer.step()

        # collect training statss 
        results = {
            "agent_{}_critic_loss".format(agent_i): vf_loss,
            "agent_{}_policy_loss".format(agent_i): pol_loss
        }
        return results

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

    def prep_training(self, device='cpu'):
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
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, 
                      rnn_policy=False, rnn_critic=False):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
            else:   # only DDPG, local critic 
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            # used in initializing agent (including policy and critic)
            agent_init_params.append({
                'num_in_pol': num_in_pol,
                'num_out_pol': num_out_pol,
                'num_in_critic': num_in_critic,
                # 'discrete_action': discrete_action,
                'rnn_policy': rnn_policy,
                'rnn_critic': rnn_critic,
            })
        # 'discrete_action' used as global to maddpg (same among agents)
        # also put it to maddpg init_dict
        init_dict = {
            'gamma': gamma, 'tau': tau, 'lr': lr,
            'hidden_dim': hidden_dim,
            'alg_types': alg_types,
            'agent_init_params': agent_init_params,
            'discrete_action': discrete_action,
        }
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
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance