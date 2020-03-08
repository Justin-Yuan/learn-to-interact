from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
import torch.distributions as D

from utils.networks import MLPNetwork, RecurrentNetwork
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.noise import OUNoise
from agents.action_selectors import DiscreteActionSelector, ContinuousActionSelector


############################################ KL scheduler

class KLCoeff(object):
    def __init__(self, kl_coeff, kl_target):
        # KL Coefficient
        self._kl_coeff = kl_coeff
        self._kl_target = kl_target
        
    def update_kl(self, sampled_kl):
        if sampled_kl > 2.0 * self._kl_target:
            self._kl_coeff *= 1.5
        elif sampled_kl < 0.5 * self._kl_target:
            self._kl_coeff *= 0.5

    def __call__(self):
        return self._kl_coeff


############################################ ppo
class PPOAgent(object):
    """
    General class for PPO agents (policy, critic, action selector, kl scheduler)
    """
    def __init__(self, self, algo_type="CCPPO", act_space=None, obs_space=None, 
                rnn_policy=False, rnn_critic=False, hidden_dim=64, lr=0.01, 
                env_obs_space=None, env_act_space=None, 
                kl_coeff=0.2, kl_target=0.01):
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

        # KL scheduler
        self.kl_coeff = KLCoeff(kl_coeff, kl_target)

        # Critic 
        self.rnn_critic = rnn_critic
        self.critic_hidden_states = None 
        
        if algo_type == "CCPPO":
            num_in_critic = 0
            for oobsp in env_obs_space:
                num_in_critic += oobsp.shape[0]
            for oacsp in env_act_space:
                # feed all acts to centralized critic
                num_in_critic += self.get_shape(oacsp)
        else:   # only DDPG, local critic 
            num_in_critic = obs_space.shape[0] + self.get_shape(act_space)

        critic_net_fn = RecurrentNetwork if rnn_critic else MLPNetwork
        self.critic = critic_net_fn(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)

        # Optimizers 
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

    
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
        # (1,H) -> (B,N,H)
        if self.rnn_policy:
            self.policy_hidden_states = self.policy.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  
        if self.rnn_critic:
            self.critic_hidden_states = self.policy.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1) 


    def compute_value(self, vf_in, h_critic=None):
        """ training critic forward with specified policy 
        Arguments:
            vf_in: (B,T,K)
        Returns:
            q: (B,T,1)
        """
        bs, ts, _ = vf_in.shape
        critic = self.critic

        if self.rnn_critic:
            q = []   # (B,1)*T
            if h_critic is None:
                h_t = self.critic_hidden_states.clone() # (B,H)
            else:
                h_t = h_critic  #.clone()

            # rollout 
            for t in range(ts):
                q_t, h_t = critic(vf_in[:,t], h_t)
                q.append(q_t)
            # (T,B,1) -> (B,T,1)
            q = torch.stack(q, 0).permute(1,0,2)   
        else:
            # (B,T,D) -> (B*T,1) -> (B,T,1)
            q, _ = critic(vf_in.reshape(bs*ts, -1)).reshape(bs,ts,1)
        return q 


    def evalaute_action(self, logit_samples, act_samples, obs, 
            h_actor=None, requires_grad=True, contract_keys=None
        ):
        """ traininsg actor forward with specified policy 
        concat all actions to be fed in critics
        Arguments:
            logit_samples: dict of (B,T,A), logits in sample
            act_samples: dict of (B,T,A), actions in sample
            obs: (B,T,O)
            requires_grad: if use _soft_act to differentiate discrete action
            contract_keys: 
                list of keys to contract dict on
                i.e. sum up all fields in log_prob, entropy, kl on given keys
        Returns:
            log_prob: action log probs (B,T,1)
            old_log_prob: action log probs with old logits (B,T,1)
            entropy: action entropy (B,T,1)
            kl: KL divergence to sample actions (B,T,1)
        """
        bs, ts, _ = obs.shape
        pi = self.policy
        log_prob_d, old_log_prob_d, entropy_d, kl_d = {}, {}, {}, {}

        # get logits for current policy
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
                    act[k].append(a)
            act = {
                k: torch.stack(ac, 0).permute(1,0,2) 
                for k, ac in act.items()
            }   # dict [(B,A)]*T -> dict (B,T,A)
        else:
            stacked_obs = obs.reshape(bs*ts, -1)  # (B*T,O)
            act, _ = pi(stacked_obs)  # act is dict of (B*T,A)
            act = {
                k: ac.reshape(bs, ts, -1)  
                for k, ac in act.items()
            }   # dict of (B,T,A)

        # make distribution and collect entities (default to all keys)
        if contract_keys is None:
            contract_keys = sorted(list(act.keys()))
        log_prob, old_log_prob, entropy, kl = 0.0, 0.0, 0.0, 0.0

        for k, seq_logits in act.items():
            if k not in contract_keys:
                continue
            action = act_samples[k]
            _, dist = self.selector.select_action(
                                seq_logits, explore=False)
            _, old_dist = self.selector.select_action(
                                logit_samples, explore=False)
            # evaluate log prob (B,T) -> (B,T,1)
            # NOTE: attention!!! if log_prob on rsample action, backprop is done twice and wrong
            log_prob += dist.log_prob(
                action.clone().detach()
            ).unsqueeze(-1)
            old_log_prob += old_dist.log_prob(action).unsqueeze(-1)
            # get current action distrib entropy
            entropy += dist.entropy().unsqueeze(-1)
            # kl between 2 distributions 
            kl += D.kl_divergence(dist, old_dist).unsqueeze(-1)

        return log_prob, old_log_prob, entropy, kl


    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        equivalent to `act` or `compute_actions`
        Arguments:
            obs: (B,O)
            explore: Whether or not to add exploration noise
        Returns:
            act_d: dict of sampled actions for this agent, (B,A)
            logits_d: dict of action logits, (B,A)
        """
        with torch.no_grad():
            logits_d, hidden_states = self.policy(obs, self.policy_hidden_states)
            self.policy_hidden_states = hidden_states   # if mlp, still defafult None

            # make distributions 
            act_d = {}
            for k, logits in logits_d.items():
                action, dist = self.selector.select_action(
                                logits, explore=explore)
                if not self.discrete_action:    # continuous action
                    action = action.clamp(-1, 1)
                act_d[k] = action
        return act_d, logits_d
    

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

