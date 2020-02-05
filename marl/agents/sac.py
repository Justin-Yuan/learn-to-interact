from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from utils.networks import MLPNetwork, RecurrentNetwork
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.noise import OUNoise




############################################ sac
class SACAgent(object):
    """
    General class for SAC agents (policy, critic1, critic2, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        pass 