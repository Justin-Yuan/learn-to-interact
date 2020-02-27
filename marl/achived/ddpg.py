from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from gym.spaces import Box, Discrete, Dict

from agents.policy import Policy, DEFAULT_ACTION
from utils.networks import MLPNetwork, RecurrentNetwork
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.noise import OUNoise


############################################ ddpg 
class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, algo_type="MADDPG", act_space=None, obs_space=None, 
                rnn_policy=False, rnn_critic=False, hidden_dim=64, lr=0.01, 
                env_obs_space=None, env_act_space=None):
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
        if isinstance(act_space, Box) or isinstance(act_space["move"], Box):
            discrete_action = False 
        elif isinstance(act_space, Discrete) or isinstance(act_space["move"], Discrete):
            discrete_action = True 
        self.discrete_action = discrete_action

        # Exploration noise 
        if not discrete_action:
            # `move`, `comm` share same continuous noise source
            self.exploration = OUNoise(self.get_shape(act_space))
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        
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
                                 constrain_out=True,
                                 discrete_action=discrete_action,
                                 rnn_policy=rnn_policy)
        self.target_policy = Policy(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action,
                                 rnn_policy=rnn_policy)
        hard_update(self.target_policy, self.policy)

        # Critic 
        self.rnn_critic = rnn_critic
        self.critic_hidden_states = None 
        
        if algo_type == "MADDPG":
            num_in_critic = 0
            for oobsp in env_observation_space:
                num_in_critic += oobsp.shape[0]
            for oacsp in env_action_space:
                # feed all acts to centralized critic
                num_in_critic += self.get_shape(oacsp)
        else:   # only DDPG, local critic 
            num_in_critic = obs_space.shape[0] + self.get_shape(act_space)

        critic_net_fn = RecurrentNetwork if rnn_critic else MLPNetwork
        self.critic = critic_net_fn(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_critic = critic_net_fn(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        hard_update(self.target_critic, self.critic)

        # Optimizers 
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)


    def get_shape(self, x, key=None):
        """ func to infer action output shape """
        if isinstance(x, Dict):
            if key is None: # sum of action space dims
                return sum([
                    x[k].n if self.discrete_action else x[k].shape[0]
                    for k in x.spaces
                ])
            elif key in x.spaces:
                return x[key].n if self.discrete_action else x[key].shape[0]
            else:   # key not in action spaces
                return 0
        else:
            return x.n if self.discrete_action else x.shape[0]
    

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale


    def init_hidden(self, batch_size):
        # (1,H) -> (B,H)
        # policy.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  
        if self.rnn_policy:
            self.policy_hidden_states = self.policy.init_hidden().expand(batch_size, -1)  
        if self.rnn_critic:
            self.critic_hidden_states = self.policy.init_hidden().expand(batch_size, -1) 


    def compute_q_val(self, vf_in, h=None, target=False):
        """ training critic forward with specified policy 
        Arguments:
            agent_i: index to agent; critic: critic network to agent; vf_in: (B,T,K);
            bs: batch size; ts: length of episode; target: if use target network
        Returns:
            q: (B*T,1)
        """
        bs, ts, _ = vf_in.shape
        critic = self.target_critic if target else self.critic

        if self.rnn_critic:
            q = []   # (B,1)*T
            h_t = self.critic_hidden_states.clone() # (B,H)
            for t in range(ts):
                q_t, h_t = critic(vf_in[:,t], h_t)
                q.append(q_t)
            q = torch.stack(q, 0).permute(1,0,2)   # (T,B,1) -> (B,T,1)
            q = q.reshape(bs*ts, -1)  # (B*T,1)
        else:
            # (B,T,D) -> (B*T,1)
            q = critic(vf_in.reshape(bs*ts, -1))
        return q 

    def compute_action(self, obs, h=None, target=False, requires_grad=True):
        """ traininsg actor forward with specified policy 
        concat all actions to be fed in critics
        Arguments:
            agent_i: index to agent; pi: policy to agent; obs: (B,T,O);
            bs: batch size; ts: length of episode; target: if use target network
        Returns:
            act: dict of (B,T,A) 
        """
        def _soft_act(x):    # x: (B,A)
            if not self.discrete_action:
                return x 
            if requires_grad:
                return gumbel_softmax(x, hard=True)
            else:
                return onehot_from_logits(x)

        bs, ts, _ = obs.shape
        pi = self.target_policy if target else self.policy

        if self.rnn_policy:
            act = []  # [(B,sum(A_k))]*T
            h_t = self.policy_hidden_states.clone() # (B,H)
            for t in range(ts):
                act_t, h_t = pi(obs[:,t], h_t)  # act_t is dict!!
                cat_act = torch.concat(
                    [_soft_act(a) for k, a in act_t.items()], -1) # (B,sum(A_k))
                act.append(cat_act) 
            act = torch.stack(act, 0).permute(1,0,2)   # (B,T,sum(A_k))
        else:
            stacked_obs = obs.reshape(bs*ts, -1)  # (B*T,O)
            act, _ = pi(stacked_obs)  # act is dict of (B*T,A)
            act = torch.concat([
                _soft_act(a).reshape(bs, ts, -1)  
                for k, a in act.items()
            ], -1)  # (B,T,sum(A_k))
        return act


    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        equivalent to `act` or `compute_actions`
        Arguments:
            obs: (B,O)
            explore: Whether or not to add exploration noise
        Returns:
            action: dict of actions for this agent, (B,A)
        """
        with torch.no_grad():
            action, hidden_states = self.policy(obs, self.policy_hidden_states)
            self.policy_hidden_states = hidden_states   # if mlp, still defafult None

            if self.discrete_action:
                for k in action:
                    if explore:
                        action[k] = gumbel_softmax(action[k], hard=True)
                    else:
                        action[k] = onehot_from_logits(action[k])
            else:  # continuous action
                idx = 0 
                noise = Variable(Tensor(self.exploration.noise()),
                                    requires_grad=False)
                for k in action:
                    if explore:
                        dim = action[k].shape[-1]
                        action[k] += noise[idx : idx+dim]
                        idx += dim 
                    action[k] = action[k].clamp(-1, 1)
        return action


    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

