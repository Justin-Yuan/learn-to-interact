import torch.nn as nn
import torch.nn.functional as F


############################################ mlp 
class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=True, 
                 use_head=True, **kwargs):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.nonlin = nonlin

        # if to use network output as base or head 
        self.use_head = use_head
        if use_head:
            self.fc3 = nn.Linear(hidden_dim, out_dim)
            if constrain_out and not discrete_action:
                # initialize small to prevent saturation
                self.fc3.weight.data.uniform_(-3e-3, 3e-3)
                self.out_fn = F.tanh
            else:  # logits for discrete action (will softmax later)
                self.out_fn = lambda x: x
                
    def init_hidden(self):
        pass 

    def forward(self, x, h=None):
        """
        Inputs:
            x (PyTorch Matrix): Batch of observations
            h: dummy hidden state
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(x)))
        h2 = self.nonlin(self.fc2(h1))
        if self.use_head:
            out = self.out_fn(self.fc3(h2))
        else:
            out = h2
        return out, h 


############################################ rnn 
class RecurrentNetwork(nn.Module):
    """
    RNN, used in policy or value function to tackle partial observability
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=True, 
                 use_head=True, **kwargs):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(RecurrentNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.hidden_dim = hidden_dim
        self.nonlin = nonlin
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

        # if to use network output as base or head 
        self.use_head = use_head
        if use_head:
            self.fc2 = nn.Linear(hidden_dim, out_dim)
            if constrain_out and not discrete_action:
                # initialize small to prevent saturation
                self.fc2.weight.data.uniform_(-3e-3, 3e-3)
                self.out_fn = F.tanh
            else:  # logits for discrete action (will softmax later)
                self.out_fn = lambda x: x

    def init_hidden(self):
        # make hidden states on same device as model (assumed batch 1) (1,H)
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, x, h_in):
        """
        Inputs:
            x (PyTorch Matrix): Batch of observations
            hidden_states (B,H): Batch of hidden states
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        x = self.nonlin(self.fc1(self.in_fn(x)))
        h = self.rnn(x, h_in)
        if self.use_head:
            out = self.out_fn(self.fc2(h))
        else:
            out = h
        return out, h


############################################ gnn 
class GraphNetwork(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True, **kwargs): 
        pass 
        