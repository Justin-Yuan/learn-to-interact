import torch.nn as nn
import torch.nn.functional as F


############################################ mlp 
class MLPNetwork(nn.Module):
    """ MLP network (can be used as value or policy)
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
    """ RNN, used in policy or value function to tackle partial observability
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



def rnn_forward_sequence(rnn, seq_inputs, seq_init_h, truncate_steps=-1):
    """ allow full bptt or truncated bptt 
    Arguments:
        rnn: RNN network 
        seq_inputs: (B,T,D)
        seq_init_h: (B,D)
        truncate_steps: int, number of bptt truncate steps 
    Returns:
        seq_qs: list of outputs, [(B,O)]*T or [dict (B,O)]*T 
    """
    def rollout(inputs, init_h):
        """ rnn partial rollout for k steps given init_h """
        hs, qs = [init_h], []   # [(B,D)]*(k+1), # [(B,O)]*k                
        for t in range(inputs.shape[1]):
            q_t, next_h = rnn(inputs[:,t], hs[-1])
            hs.append(next_h)
            qs.append(qs)
        return hs, qs

    # RNN forward run 
    if truncate_steps <= 0:  # normal bptt 
        _, seq_qs = rollout(seq_inputs, seq_init_h)
        return seq_qs
    else:   # truncated bptt 
        with torch.no_grad():   # pre-run to get all hidden states
            seq_hs, _ = rollout(seq_inputs, seq_init_h)
        # re-run each truncated sub-sequences (for proper backprop)
        seq_qs = []
        for t in range(1,seq_inputs.shape[1]+1):
            idx = 0 if t <= truncate_steps else t - truncate_steps
            t_inputs = seq_inputs[:,idx:t]  # (B,truncate_steps,D)
            t_init_h = seq_hs[idx]  # (B,D)
            _, t_qs = rollout(t_inputs, t_init_h)
            seq_qs.append(t_qs[-1])
        return seq_qs


############################################ gcn 
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, nonlin=F.relu,
                constrain_out=False, norm_in=False, discrete_action=True, 
                use_head=True, **kwargs):
        pass 

    def forward(self, nodes, edges, masks=None):
        pass 



############################################ gnn 
class GraphNetwork(nn.Module):
    """ Generic graph net. 
    reference: https://github.com/lrjconan/LanczosNetwork --> GAT model
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=True, 
                 use_head=True, num_edgetype=1, num_layer=2, num_heads=1, 
                 dropout=0.0, output_level="graph", **kwargs): 
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(GraphNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.hidden_dim = hidden_dim
        self.nonlin = nonlin
        self.num_layer = num_layer
        self.num_edgetype = num_edgetype
        self.dropout = dropout
        self.output_level = output_level   # graph or node

        # Input embedding layer 
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Propagatoin layers 
        self.dim_list = [hidden_dim]*(self.num_layer+1)
        self.num_heads = [num_heads]*self.num_layer

        self.filter = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(
                        dim_list[tt] *
                        (int(tt == 0) + int(tt != 0) * self.num_heads[tt] * self.num_edgetype),
                        dim_list[tt + 1], 
                        bias=False
                    ) for _ in range(self.num_heads[tt])     # heads 
                ]) for _ in range(self.num_edgetype)       # edge types
            ]) for tt in range(self.num_layer)      # layers
        ])      # 1st layer output concat all heads, so for first layer, input is not concat

        # Attention layers 
        self.att_net_1 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(dim_list[tt + 1], 1)
                    for _ in range(self.num_heads[tt])
                ]) for _ in range(self.num_edgetype)
            ]) for tt in range(self.num_layer)
        ])

        self.att_net_2 = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(dim_list[tt + 1], 1)
                    for _ in range(self.num_heads[tt])
                ]) for _ in range(self.num_edgetype)
            ]) for tt in range(self.num_layer)
        ])

        # Biases
        self.state_bias = [
            [[None] * self.num_heads[tt]] * self.num_edgetype 
            for tt in range(self.num_layer)
        ]
        for tt in range(self.num_layer):
            for jj in range(self.num_edgetype):
                for ii in range(self.num_heads[tt]):
                    self.state_bias[tt][jj][ii] = torch.nn.Parameter(
                                            torch.zeros(dim_list[tt + 1]))
                    self.register_parameter('bias_{}_{}_{}'.format(ii, jj, tt),
                                            self.state_bias[tt][jj][ii])

        # if to use network output as base or head 
        self.use_head = use_head
        if use_head:
            self.output_func = nn.Linear(hidden_dim, out_dim)
            if constrain_out and not discrete_action:
                # initialize small to prevent saturation
                self.output_func.weight.data.uniform_(-3e-3, 3e-3)
                self.out_fn = F.tanh
            else:  # logits for discrete action (will softmax later)
                self.out_fn = lambda x: x


    def forward(self, nodes, edges, masks=None, h_in=None):
        """ stack of gcn layers with attention 
        Args:
            - nodes: (B, N, I)
            - edges: (B, N, N, E)
            - masks: (B, N)
        Returns:
            - out: (B, N, O) or (B, O) depending on `output_level`
        """
        batch_size, num_node, _ = node_feat.shape
        state = self.embedding(nodes)  # (B, N, D)

        for tt in range(self.num_layer):
            h = []

            # transform & aggregate features
            for jj in range(self.num_edgetype):
                for ii in range(self.num_heads[tt]):
                    
                    # transformed features
                    state_head = F.dropout(state, self.dropout, training=self.training)
                    Wh = self.filter[tt][jj][ii](
                        state_head.view(batch_size * num_node, -1)
                    ).view(batch_size, num_node, -1)  # (B, N, D)

                    # attention weights
                    att_weights_1 = self.att_net_1[tt][jj][ii](Wh)  # (B, N, 1)
                    att_weights_2 = self.att_net_2[tt][jj][ii](Wh)  # (B, N, 1)
                    att_weights = att_weights_1 + att_weights_2.transpose(1, 2)  # (B, N, N) dense matrix
                    att_weights = F.softmax(
                        F.leaky_relu(att_weights, negative_slope=0.2) + edges[:, :, :, jj],
                        dim=1)
                    
                    # dropout attn weights and features
                    att_weights = F.dropout(
                        att_weights, self.dropout, training=self.training)  # (B, N, N)
                    Wh = F.dropout(Wh, self.dropout, training=self.training)  # (B, N, D)

                    # aggregation step
                    msg = torch.bmm(att_weights, Wh) + self.state_bias[tt][jj][ii].view(1, 1, -1)
                    if tt == self.num_layer - 1:
                        msg = F.elu(msg)
                    h += [msg]  # (B, N, D)

            # propagation step 
            if tt == self.num_layer - 1:
                state = torch.mean(torch.stack(h, dim=0), dim=0)  # (B, N, D), average all heads & edges
            else:
                state = torch.cat(h, dim=2)     # (B, N, D * #edge_types * #heads)

        # output
        out = self.output_func(
            state.view(batch_size * num_node, -1)
        ).view(batch_size, num_node, -1)    # (B, N, O)
        # if output is `graph-level`, out is now (B, N, 1), convert to (B, 1)
        if self.output_level == "graph":
            if masks is not None:
                out = out.squeeze() * masks   # (B, N)
            out = torch.mean(out, dim=1)    # simple sum, could extend to weighted sum (attention)

        return out 
        

############################################ google graph net 

class GraphNet(nn.Module):
    def __init__(self):
        pass 

    def forward(self, nodes, edges, masks=None):
        pass 


