import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, compute_advantages, explained_variance_torch
from utils.agents import PPOAgent



MSELoss = torch.nn.MSELoss()

class CCPPO(object):
    """
    Wrapper class for PPO-esque (i.e. also CCPPO) agents in multi-agent task
    """
    def __init__(self, agent_init_params=None, alg_types=None,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                #  discrete_action=False
                use_gae=True, lambda_=1.0, clip_param=0.3, 
                vf_clip_param=10.0, vf_loss_coeff=1.0, 
                entropy_coeff=0.1, kl_coeff=0.2, kl_target=0.01, 
                **kwargs
        ):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (PPO
                                       or CCPPO)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space

            # ppo specific 
            # reference: https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/ppo.py
            - vf_clip_param (float): Clip param for the value function. Note that this is sensitive to the
                scale of the rewards. If your expected V is large, increase this.
            - kl_coeff (float): Initial coefficient for KL divergence.
            - kl_target (float): Target value for KL divergence.
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [PPOAgent(lr=lr, hidden_dim=hidden_dim, **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr

        # ppo specific params 
        self.use_gae = use_gae
        self.lambda_ = lambda_
        self.clip_param = clip_param
        self.vf_clip_param = vf_clip_param
        self.vf_loss_coeff = vf_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.kl_coeff = kl_coeff
        self.kl_target = kl_target

        # self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
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
            a.critic.train()
            
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = a.policy.to(device)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = a.critic.to(device)
            self.critic_dev = device

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
    def init_from_env(
        cls, env, agent_alg="CCPPO", adversary_alg="CCPPO",
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
        # also put agent init dicts to ccppo init_dict
        init_dict = {
            'gamma': gamma, 'tau': tau, 'lr': lr,
            'hidden_dim': hidden_dim,
            'alg_types': alg_types,
            'agent_init_params': agent_init_params,
            # 'discrete_action': discrete_action,
        }
        # ppo specific configs
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
            observations: [(B,O)]*N, List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Returnss:
            actions: List of action (np array or dict of it) for each agent
            info: dict of info needed for decentralized/centralized training
        """
        actions, info = [], {}
        logits, values = [], []

        # collect & evaluate actions
        for a, obs in zip(self.agents, observations):
            act_d, logits_d = a.step(obs, explore=explore) 
            actions.append(act_d)   # dict (B,A)
            logits.append(logits_d)    # dict (B,A)
        info["logits"] = logits  # [dict (B,A)]*N
        
        return actions, info

    ############################################ NOTE: update
    def init_hidden(self, batch_size):
        """ for rnn policy, training or evaluation """
        for a in self.agents:
            a.init_hidden(batch_size) 

    def build_full_obs(self, obs, next_obs):
        """ append last obervation to obs for full obs sequence
        Arguments:
            obs, next_obs: [(B,T,O)]*N or [dict (B,T,O)]*N
        Returns:
            full_obs: [(B,T+1,O)]*N or [dict (B,T+1,O)]*N
        """
        full_obs = []
        for ob, nob in zip(obs, next_obs):
            if isinstance(ob, dict):
                fob = {
                    k: torch.cat([
                        obs[k], next_obs[k][:,-1].unsqueeze(1) 
                    ], 1) for k in obs        # (B,T+1,O)
                }
            else:
                last_ob = next_obs[:,-1].unsqueeze(1)   # (B,1,O)
                fob = torch.cat([obs, last_ob], 1)  # (B,T+1,O)

            full_obs.append(fob)
        return full_obs


    def prepare_samples(self, samples):
        """ compute advantages and value targets
        Arguments:
            samples: obs, acs, rews, next_obs, dones, logits
                [(B,T,D)]*N, obs, next_obs, action can be [dict (B,T,D)]*N
        Returns:
            advantages: n-step returns or advantage, (B,T,1) 
            vf_preds: prediction for value function, (B,T,1)
        """
        obs, acs, rews, next_obs, dones, old_logits = sample # each [(B,T,D)]*N 
        advantages, vf_preds = [], []
        
        for agent_i, a in enumerate(self.agents):
            # construct value predictions
            # [(B,T+1,O)]*N or [dict (B,T+1,O)]*N
            full_obs = self.build_full_obs(obs, next_obs)   

            if self.alg_types[agent_i] == 'CCPPO'
                vf_in = torch.cat([
                    *self.flatten_obs(full_obs, ma=True), 
                ], dim=-1)  # [(B,T+1,O)]*N -> (B,T+1,O*N) 
            else:  # PPO
                vf_in = self.flatten_obs(full_obs[agent_i]) # (B,T+1,O)
            
            full_vf_preds = a.compute_value(vf_in) # (B,T+1,1)
            # if done, last_r is 0, otherwise use last value pred (for last next obs)
            last_r = (1.0 - dones[agent_i][:,-1]) * full_vf_preds[:,-1] # (B,1)

            # advantages and value targets
            advtg, value_targets = compute_advantages(
                last_r, rews[agent_i], full_vf_preds[:,:-1], 
                gamma=self.gamma, lambda_=self.lambda_, use_gae=self.use_gae, 
                use_critic=True, norm_advantages=True
            )
            advantages.append(advtg)
            vf_preds.append(value_targets)

        return advantages, vf_preds


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
            acs: (B,T,A), or dict of (B,T,A), or list []*N of it 
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

    def update(self, sample, agent_i, parallel=False, grad_norm=0.5, contract_keys=None):
        """ Update parameters of agent model based on sample from replay buffer
        Arguments:
            sample: [(B,T,D)]*N, obs, next_obs, action can be [dict (B,T,D)]*N
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
        """
        # each is [(B,T,D)]*N 
        obs, acs, rews, next_obs, dones, old_logits, advantages, vf_preds = sample 
        bs, ts, _ = obs[agent_i].shape 
        self.init_hidden(bs)  # use pre-defined init hiddens 
        curr_agent = self.agents[agent_i]

        # NOTE: Critic update
        curr_agent.critic_optimizer.zero_grad()

        # value func
        if self.alg_types[agent_i] == 'CCPPO':
            # [(B,T,O)_i, ...] -> (B,T,O*N)
            vf_in = torch.cat([
                *self.flatten_obs(obs, ma=True), 
            ], dim=-1)
        else:  # PPO
            vf_in = self.flatten_obs(obs[agent_i]) # (B,T,O)
        actual_value = curr_agent.compute_value(vf_in) # (B,T,1)
        
        # bellman errors (PPO clipped style)
        vf_loss1 = (actual_value - vf_preds) ** 2
        vf_clipped = vf_preds + (actual_value - vf_preds).clamp(
                                -self.vf_clip_param, self.vf_clip_param)
        vf_loss2 = (vf_clipped - vf_preds) ** 2
        vf_loss = torch.max(vf_loss1, vf_loss2).mean()

        critic_loss = self.vf_loss_coeff * vf_loss
        critic_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        if grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), grad_norm)
        curr_agent.critic_optimizer.step()


        # NOTE: Policy update
        curr_agent.policy_optimizer.zero_grad()
        
        # ppo policy update 
        act_eval_out = curr_agent.evalaute_action(
            old_logits[agent_i], acs[agent_i], obs[agent_i], 
            contract_keys=contract_keys
        ) # all (B,T,1)
        curr_log_prob, old_log_prob, entropy, kl = act_eval_out

        logp_ratio = torch.exp(curr_log_probs - old_log_probs)
        policy_loss = -torch.min(
            advantages * logp_ratio, 
            advantages * logp_ratio.clamp(1-self.clip_param, 1+self.clip_param)
        )   # (B,T,1)
        policy_loss = policy_loss.mean()

        # kl loss on current & previous policy outputs
        kl_loss = kl.mean()
        # update kl coefficient per update (with mean/expected kl)
        curr_agent.kl_coeff.update_kl(kl_loss)

        # entropy loss on current policy outputs
        entropy_loss = entropy.mean()

        actor_loss = policy_loss
        actor_loss += curr_agent.kl_coeff() * kl_loss
        actor_loss += self.entropy_coeff * entropy_loss
        actor_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        if grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), grad_norm)
        curr_agent.policy_optimizer.step()

        # NOTE: collect training statss 
        results = {}
        key_list = [
            "critic_loss", 
            "policy_loss", 
            "kl_loss", 
            "entropy_loss",
            "explained_variance"
        ]
        val_list = [
            vf_loss,
            policy_loss,
            kl_loss,
            entropy_loss,
            explained_variance(vf_preds, actual_value)
        ]

        for k, v in zip(key_list, val_list):
            key = "agent_{}/{}".format(agent_i, k)
            value = v.data.cpu().numpy()
            results[key] = value
            self.agent_losses[key].append(value)        
        return results

    

    