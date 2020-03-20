import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.networks import MLPNetwork, RecurrentNetwork

DEFAULT_ACTION = "default"

############################################ policy
class Policy(nn.Module):
    """ wrapper on agent networks 
    supports multiple action outputs, e.g. (move, comm)
    """
    def __init__(self, num_in_pol=0, num_out_pol=0, hidden_dim=64, 
                constrain_out=False, norm_in=False, discrete_action=True, 
                rnn_policy=False, **kwargs):
        """
        Arguments:
            num_in_pol (int): policy input size
            num_out_pol (int or dict): policy output specs
        """
        super(Policy, self).__init__()
        policy_net_fn = RecurrentNetwork if rnn_policy else MLPNetwork

        if isinstance(num_out_pol, dict):   # multiple outputs
            self.multi_output = True 
            self.base = policy_net_fn(num_in_pol, hidden_dim,
                                    hidden_dim=hidden_dim,
                                    use_head=False)
            idx2head, heads = {}, []
            for i, (head_name, out_dim) in enumerate(num_out_pol.items()):
                idx2head[i] = head_name
                heads.append(nn.Linear(hidden_dim, out_dim))
            self.idx2head = idx2head
            self.heads = nn.ModuleList(heads)

            if constrain_out and not discrete_action:
                # initialize small to prevent saturation
                for i in range(len(self.idx2head)):
                    self.heads[i].weight.data.uniform_(-3e-3, 3e-3)
                # heads share out_fn for now 
                self.out_fn = F.tanh
            else:  # logits for discrete action (will softmax later)
                self.out_fn = lambda x: x

        else:   # single output
            self.multi_output = False
            self.base_head = policy_net_fn(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                #  constrain_out=True,
                                 constrain_out=constrain_out,
                                 norm_in=norm_in,
                                 discrete_action=discrete_action,
                                 use_head=True)
        
    def forward(self, x, h=None, **kwargs):
        """ output is dict of actions and hidden, 
        if not `multi_output`, action use default key `DEFAULT_ACTION`,
        if mlp policy, hidden is unused, return None still
        Returns:
            act_dict: dict of action (B,A)
            h_out: output hidden state (B,H)
        """
        act_dict = {}
        if self.multi_output:
            out, h_out = self.base(x, h)
            for i, k in self.idx2head.items():
                act = self.out_fn(self.heads[i](out))
                act_dict[k] = act
        else:
            act, h_out = self.base_head(x, h)
            act_dict[DEFAULT_ACTION] = act
        return act_dict, h_out

    def init_hidden(self):
        # only if `rnn_policy` is enabled, chained to `init_hidden` to rnn net
        if self.multi_output:
            return self.base.init_hidden()
        else:
            return self.base_head.init_hidden()


