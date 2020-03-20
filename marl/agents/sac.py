from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from gym.spaces import Box, Discrete, Dict

from agents.policy import Policy, DEFAULT_ACTION
from utils.networks import MLPNetwork, RecurrentNetwork
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.noise import OUNoise
from agents.action_selectors import DiscreteActionSelector, ContinuousActionSelector



############################################ Automated Temperature Adjustment

class LogAlpha(nn.Module):
    """ log of temperature parameter 
    """
    def __init__(self, init_value=0.0):
        super(LogAlpha, self).__init__()
        self.log_alpha = nn.Parameter(torch.tensor(init_value, requires_grad=True))
        
    def get_alpha(self):
        return torch.exp(self.log_alpha)

    def __call__(self):
        return self.log_alpha


############################################ sac 
class SACAgent(object):
    """
    General class for SAC agents (policy, critic1, critic2, target critic1 , target
    critic2, log_alpha, action selector)
    """
    def __init__(self, algo_type="MASAC", act_space=None, obs_space=None, 
                rnn_policy=False, rnn_critic=False, hidden_dim=64, lr=0.01, 
                env_obs_space=None, env_act_space=None, **kwargs):
        """
        Inputs:
            act_space: single agent action space (single space or Dict)
            obs_space: single agent observation space (single space Dict)
        """
        self.algo_type = algo_type
        self.act_space = act_space 
        self.obs_space = obs_space

        # continuous or discrete action (only look at `move` action, assume
        # move and comm space both discrete or continuous)
        tmp = act_space.spaces["move"] if isinstance(act_space, Dict) else act_space
        self.discrete_action = False if isinstance(tmp, Box) else True 
        
        # Policy (supports multiple outputs)
        self.rnn_policy = rnn_policy
        self.policy_hidden_states = None 

        num_in_pol = obs_space.shape[0]
        if isinstance(act_space, Dict):
            # hard specify now, could generalize later 
            num_out_pol = {
                "move": self.get_shape(act_space, "move"), 
                "comm": self.get_shape(act_space, "comm")
            }
        else:
            num_out_pol = self.get_shape(act_space)

        self.policy = Policy(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                #  constrain_out=True,
                                 discrete_action=self.discrete_action,
                                 rnn_policy=rnn_policy)

        # action selector (distribution wrapper)
        if self.discrete_action:
            self.selector = DiscreteActionSelector()
        else:
            self.selector = ContinuousActionSelector()

        # Critic 
        self.rnn_critic = rnn_critic
        self.critic_hidden_states = None 
        
        if algo_type == "MASAC":
            num_in_critic = 0
            for oobsp in env_obs_space:
                num_in_critic += oobsp.shape[0]
            for oacsp in env_act_space:
                # feed all acts to centralized critic
                num_in_critic += self.get_shape(oacsp)
        else:   # only DDPG, local critic 
            num_in_critic = obs_space.shape[0] + self.get_shape(act_space)

        critic_net_fn = RecurrentNetwork if rnn_critic else MLPNetwork
        self.critic1 = critic_net_fn(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.critic2 = critic_net_fn(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_critic1 = critic_net_fn(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        self.target_critic2 = critic_net_fn(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        hard_update(self.target_critic1, self.critic1)
        hard_update(self.target_critic2, self.critic2)

        # alpha
        self.log_alpha = LogAlpha(0.0)

        # Optimizers 
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=lr)
        self.alpha_optimizer = Adam((self.log_alpha,), lr=lr)


    def get_shape(self, x, key=None):
        """ func to infer action output shape """
        if isinstance(x, Dict):
            if key is None: # sum of action space dims
                return sum([
                    x.spaces[k].n if self.discrete_action else x.spaces[k].shape[0]
                    for k in x.spaces
                ])
            elif key in x.spaces:
                return x.spaces[key].n if self.discrete_action else x.spaces[key].shape[0]
            else:   # key not in action spaces
                return 0
        else:
            return x.n if self.discrete_action else x.shape[0]

    def init_hidden(self, batch_size):
        # (1,H) -> (B,H)
        # policy.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  
        if self.rnn_policy:
            self.policy_hidden_states = self.policy.init_hidden().expand(batch_size, -1)  
        if self.rnn_critic:
            self.critic1_hidden_states = self.critic1.init_hidden().expand(batch_size, -1) 
            self.critic2_hidden_states = self.critic2.init_hidden().expand(batch_size, -1) 


    def compute_value(self, vf_in, h1_critic=None, h2_critic=None, target=False):
        """ training critic forward with specified policy 
        Arguments:
            vf_in: (B,T,K)
            target: if use target network
        Returns:
            q1, q2: (B*T,1)
        """
        bs, ts, _ = vf_in.shape
        critic1 = self.target_critic1 if target else self.critic1
        critic2 = self.target_critic2 if target else self.critic2

        if self.rnn_critic:
            q1, q2 = [], []   # (B,1)*T
            if h_critic is None:
                h1_t = self.critic1_hidden_states.clone() # (B,H)
                h2_t = self.critic2_hidden_states.clone() 
            else:
                h1_t = h1_critic  #.clone()
                h2_t = h2_critic

            # rollout 
            for t in range(ts):
                q1_t, h1_t = critic1(vf_in[:,t], h1_t)
                q1.append(q1_t)
                q2_t, h2_t = critic2(vf_in[:,t], h2_t)
                q2.append(q2_t)

            q1 = torch.stack(q1, 0).permute(1,0,2)   # (T,B,1) -> (B,T,1)
            q1 = q1.reshape(bs*ts, -1)  # (B*T,1)
            q2 = torch.stack(q2, 0).permute(1,0,2)   
            q2 = q2.reshape(bs*ts, -1)  
        else:
            # (B,T,D) -> (B*T,1)
            q1, _ = critic1(vf_in.reshape(bs*ts, -1))
            q2, _ = critic2(vf_in.reshape(bs*ts, -1))
        return q1, q2


    def compute_action_logprob(self, obs, h_actor=None, reparameterize=True):
        """ traininsg actor forward with specified policy 
        concat all actions to be fed in critics
        Arguments:
            obs: (B,T,O)
            target: if use target network
            reparameterize: if to use reparameterization in action selection
        Returns:
            act_d: dict of sampled actions (B,T,A) 
            log_prob_d: dict of action log probs (B,T,1)
            act/logits_d: dict of output logits to sample from (B,T,A)
        """
        bs, ts, _ = obs.shape
        pi = self.policy
        act_d, log_prob_d = {}, {}

        if self.rnn_policy:
            act = defaultdict(list)
            if h_actor is None:
                h_t = self.policy_hidden_states.clone() # (B,H)
            else:
                h_t = h_actor   #.clone()

            # rollout 
            for t in range(ts):
                act_t, h_t = pi(obs[:,t], h_t)  # act_t is dict (B,A)
                for k, a in act_t.items():
                    act[k].append(_soft_act(a))
            act = {
                k: torch.stack(ac, 0).permute(1,0,2) 
                for k, ac in act.items()
            }   # dict [(B,A)]*T -> dict (B,T,A)
        else:
            stacked_obs = obs.reshape(bs*ts, -1)  # (B*T,O)
            act, _ = pi(stacked_obs)  # act is dict of (B*T,A)
            act = {
                k: _soft_act(ac).reshape(bs, ts, -1)  
                for k, ac in act.items()
            }   # dict of (B,T,A)

        # make distribution & evaluate log probs
        for k, seq_logits in act.items():
            action, dist = self.selector.select_action(
                                seq_logits, explore=False, 
                                reparameterize=reparameterize)
            if not self.discrete_action:    # continuous action
                action = action.clamp(-1, 1)
            act_d[k] = action   # (B,T,A)
            # evaluate log prob (B,T) -> (B,T,1)
            # NOTE: attention!!! if log_prob on rsample action, backprop is done twice and wrong
            log_prob_d[k] = dist.log_prob(
                action.clone().detach()
            ).unsqueeze(-1)

        return act_d, log_prob_d, act


    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        equivalent to `act` or `compute_actions`
        Arguments:
            obs: (B,O)
            explore: Whether or not to add exploration noise
        Returns:
            act_d: dict of actions for this agent, (B,A)
            log_prob_d: dict of action log probs, (B,1)
        """
        with torch.no_grad():
            logits_d, hidden_states = self.policy(obs, self.policy_hidden_states)
            self.policy_hidden_states = hidden_states   # if mlp, still defafult None

            # make distributions 
            act_d, log_prob_d = {}, {}
            for k, logits in logits_d.items():
                action, dist = self.selector.select_action(
                                logits, explore=explore)
                if not self.discrete_action:    # continuous action
                    action = action.clamp(-1, 1)
                act_d[k] = action
                # get log prob of sampled action
                log_prob_d[k] = dist.log_prob(action).unsqueeze(-1) # (B,1)
        return act_d, log_prob_d


    def get_params(self):
        return {
            'policy': self.policy.state_dict(),
            'critic1': self.critic.state_dict(),
            'critic2': self.critic.state_dict(),
            'log_alpha': self.log_alpha.detach().item(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict()
        }

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic1.load_state_dict(params['critic1'])
        self.critic2.load_state_dict(params['critic2'])
        with torch.no_grad():
            self.log_alpha[:] = state_dict["log_alpha"]
        self.target_critic1.load_state_dict(params['target_critic1'])
        self.target_critic2.load_state_dict(params['target_critic2'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic1_optimizer.load_state_dict(params['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(params['critic2_optimizer'])
        self.alpha_optimizer.load_state_dict(params['alpha_optimizer'])

