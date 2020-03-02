from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from utils.networks import MLPNetwork, RecurrentNetwork
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.noise import OUNoise

############################################ ppo
class PPOAgent(object):
    """
    General class for PPO agents (policy, critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True, rnn_policy=False, rnn_critic=False):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.rnn_policy = rnn_policy
        self.rnn_critic = rnn_critic
        self.policy_hidden_states = None 
        self.critic_hidden_states = None 

        policy_net_fn = RecurrentNetwork if rnn_policy else MLPNetwork
        critic_net_fn = RecurrentNetwork if rnn_policy else MLPNetwork
        self.policy = policy_net_fn(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic = critic_net_fn(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def init_hidden(self, batch_size):
        # (1,H) -> (B,N,H)
        if self.rnn_policy:
            self.policy_hidden_states = self.policy.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  
        if self.rnn_critic:
            self.critic_hidden_states = self.policy.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1) 

    def evaluate_actions(self, obs, ):
        return 

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        if self.rnn_policy:
            action, hidden_states = self.policy(obs, self.policy_hidden_states)
            self.policy_hidden_states = hidden_states
        else:
            action = self.policy(obs)
            
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action
    
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

