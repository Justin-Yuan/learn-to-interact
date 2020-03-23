import torch.nn as nn
import torch.nn.functional as F


#####################################################################################
### mlp
#####################################################################################
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


#####################################################################################
### rnn
#####################################################################################
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



def rollout(rnn, inputs, init_h):
    """ rnn partial rollout for k steps given init_h 
    Arguments:
        rnn: RNN network 
        inputs: (B,T,D)
        init_h: (B,D)
    Returns:
        hs: list of hidden states, [(B,H)]*T 
        qs: list of outputs, [(B,O)]*T or [dict (B,O)]*T
    """
    hs, qs = [init_h], []   # [(B,D)]*(k+1), # [(B,O)]*k                
    for t in range(inputs.shape[1]):
        q_t, next_h = rnn(inputs[:,t], hs[-1])
        hs.append(next_h)
        qs.append(q_t)
    return hs, qs


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
    # RNN forward run 
    if truncate_steps <= 0:  # normal bptt 
        _, seq_qs = rollout(rnn, seq_inputs, seq_init_h)
        return seq_qs
    else:   # truncated bptt 
        with torch.no_grad():   # pre-run to get all hidden states
            seq_hs, _ = rollout(rnn, seq_inputs, seq_init_h)
        # re-run each truncated sub-sequences (for proper backprop)
        seq_qs = []
        for t in range(1,seq_inputs.shape[1]+1):
            idx = 0 if t <= truncate_steps else t - truncate_steps
            t_inputs = seq_inputs[:,idx:t]  # (B,truncate_steps,D)
            t_init_h = seq_hs[idx]  # (B,D)
            _, t_qs = rollout(rnn, t_inputs, t_init_h)
            seq_qs.append(t_qs[-1])
        return seq_qs


#####################################################################################
### attention
#####################################################################################
class AttentionLayer(nn.Module):
    """ single head attention on input vectors 
    """
    def __init__(self, input_dim, hidden_dim=64, nonlin=F.relu, **kwargs):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(AttentionLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.nonlin = nonlin
        self.attn_fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.attn_fc2 = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x, exclude_self=True):
        """ x: (B,N,D)
        """
        b, n, d = x.shape
        h1 = self.attn_fc1(x)  # (B,N,H)
        h2 = self.attn_fc2(x)
        # (B,N,H) x (B,H,N) -> (B,N,N)
        attn_weights = torch.bmm(h1, h2.permute(0,2,1))
        # if to set its own attn logit to -infinity 
        if exclude_self:
            mask = torch.ones(n,n) - torch.eye(n) - torch.eye(float("Inf"))
            mask = mask.unsqueeze(0).expand(b,n,n)
            att_weights = att_weights * mask
        # normalize 
        attn_weights = F.softmax(att_weights, -1)
        # linear combo, (B,N,N) x (B,N,H) -> (B,N,H)
        out = torch.bmm(att_weights, x) 
        return out 



#####################################################################################
### gcn
#####################################################################################
class GCN(nn.Module):
    """ graph net with simple graph conv. 
    reference: https://github.com/lrjconan/LanczosNetwork --> GCN model
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64, nonlin=F.elu,
                constrain_out=False, norm_in=False, discrete_action=True, 
                use_head=True, num_edgetype=1, num_layer=2,
                dropout=0.0, output_level="graph", **kwargs):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(GCN, self).__init__() 

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
        # in - hid - hid - hid - out
        dim_list = [hidden_dim]*(self.num_layer+1)
        self.dim_list = dim_list

        self.filter = nn.ModuleList([
            nn.Linear(dim_list[tt] * self.num_edgetype, dim_list[tt + 1])
            for tt in range(self.num_layer)
        ])

        # if to use network output as base or head 
        self.use_head = use_head
        if use_head:
            # self.output_func = nn.Linear(hidden_dim, out_dim)
            self.output_func = nn.Linear(dim_list[-1], out_dim)
            if constrain_out and not discrete_action:
                # initialize small to prevent saturation
                self.output_func.weight.data.uniform_(-3e-3, 3e-3)
                self.out_fn = F.tanh
            else:  # logits for discrete action (will softmax later)
                self.out_fn = lambda x: x

        # attention
        self.att_func = nn.Sequential(*[nn.Linear(dim_list[-1], 1), nn.Sigmoid()])
        self._init_param()

    def _init_param(self):
        """ per component xavier initializations """
        for ff in self.filter:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()

        if self.use_head:
            ff = self.output_func 
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()

        for ff in self.att_func:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()

    def forward(self, nodes, edges, masks=None, h_in=None):
        """ stack of gcn layers with attention 
        Args:
            - nodes: (B, N, I)
            - edges: (B, N, N, E)
            - masks: (B, N, 1)
        Returns:
            - out: (B, N, O) or (B, O) depending on `output_level`
        """
        batch_size, num_node, _ = node_feat.shape
        state = self.embedding(nodes)  # (B,N,D)
        # extend edges (B,N,N) -> (B,N,N,1)
        if len(edge_types.shape) == 3:  
            edges = edges.unsqueeze(-1)

        # propagation
        for tt in range(self.num_layer):
            msg = []

            for ii in range(self.num_edgetype):
                msg += [torch.bmm(edges[:, :, :, ii], state)]  # (B,N,D)

            msg = torch.cat(msg, dim=2).view(batch_size * num_node, -1)
            state = self.nonlin(self.filter[tt](msg)).view(batch_size, num_node, -1)
            state = F.dropout(state, self.dropout, training=self.training)

        # output
        out = self.output_func(
            state.view(batch_size * num_node, -1)
        ).view(batch_size, num_node, -1)    # (B, N, O)

        # masking 
        if masks is not None:
            if len(masks.shape) == 2:
                masks = masks.unsqueeze(-1).expand_as(out) # (B,N,O)
            out = out * masks   # (B,N,O)

        # if output is `graph-level`, out is now (B, N, O), convert to (B, O)
        if self.output_level == "graph":
            att_weight = self.att_func(
                state.view(batch_size * num_node, -1)
            ).view(batch_size, num_node, -1)  # (B,N,1)
            out = torch.mean(att_weight * out, dim=1)    # (B,O), could use simple average instead 

        return out, h_in


#####################################################################################
### gat
#####################################################################################
class GAT(nn.Module):
    """ graph net with attention. 
    reference: https://github.com/lrjconan/LanczosNetwork --> GAT model
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.elu,
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
        super(GAT, self).__init__()

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
        # in - hid - hid - hid - out
        dim_list = [hidden_dim]*(self.num_layer+1)
        self.dim_list = dim_list
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
            # self.output_func = nn.Linear(hidden_dim, out_dim)
            self.output_func = nn.Linear(dim_list[-1], out_dim)
            if constrain_out and not discrete_action:
                # initialize small to prevent saturation
                self.output_func.weight.data.uniform_(-3e-3, 3e-3)
                self.out_fn = F.tanh
            else:  # logits for discrete action (will softmax later)
                self.out_fn = lambda x: x

        # attention
        self.att_func = nn.Sequential(*[nn.Linear(dim_list[-1], 1), nn.Sigmoid()])


    def forward(self, nodes, edges, masks=None, h_in=None):
        """ stack of gcn layers with attention 
        Args:
            - nodes: (B, N, I)
            - edges: (B, N, N, E)
            - masks: (B, N, 1)
        Returns:
            - out: (B, N, O) or (B, O) depending on `output_level`
        """
        batch_size, num_node, _ = node_feat.shape
        state = self.embedding(nodes)  # (B, N, D)
        # extend edges (B,N,N) -> (B,N,N,1)
        if len(edge_types.shape) == 3:  
            edges = edges.unsqueeze(-1)

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

                    # propagation step
                    msg = torch.bmm(att_weights, Wh) + self.state_bias[tt][jj][ii].view(1, 1, -1)
                    if tt != self.num_layer - 1:
                        msg = self.nonlin(msg)
                    h += [msg]  # (B, N, D)

            # aggregation step 
            if tt == self.num_layer - 1:
                state = torch.mean(torch.stack(h, dim=0), dim=0)  # (B, N, D), average all heads & edges
            else:
                state = torch.cat(h, dim=2)     # (B, N, D * #edge_types * #heads)

        # output (if not use head, output is an intermediatee representation)
        out = self.output_func(
            state.view(batch_size * num_node, -1)
        ).view(batch_size, num_node, -1)    # (B, N, O)

        # masking 
        if masks is not None:
            if len(masks.shape) == 2:
                masks = masks.unsqueeze(-1).expand_as(out) # (B,N,O)
            out = out * masks   # (B,N,O)

        # if output is `graph-level`, out is now (B, N, O), convert to (B, O)
        if self.output_level == "graph":
            att_weight = self.att_func(
                state.view(batch_size * num_node, -1)
            ).view(batch_size, num_node, -1)  # (B,N,1)
            out = torch.mean(att_weight * out, dim=1)    # (B,O), could use simple average instead 

        return out, h_in
        

#####################################################################################
### google graph net 
#####################################################################################

class GraphTuple(object):
    """ simple wrapper for batach of graphs
    """
    def __init__(self, nodes, edges, global_u=None, 
        edge_rs=None, node_masks=None, edge_masks=None
    ):
        """ Inputs: 
        - nodes: (B,Nv,Dv) node attributes
        - edges: (B,Ne,De), edge attributes
        - global_u: (B,Du), global node attribute
        - edge_rs: (B,Ne,2), list of receiver & sender indices
        - node_masks: (B,Nv,1) 
        - edge_masks: (B,Ne,1)
        """
        self.nodes = nodes 
        sself.edges = edges 
        self.global_u = global_u
        self.edge_rs = edge_rs
        self.node_masks = node_masks
        self.edge_masks = edge_masks

        # size info 
        self.batch_size, self.n_nodes, self.node_dim = nodes.shape
        _, self.n_edges, self.edge_dim = edges.shape
        self.u_dim = global_u.shape[-1] if global_u is not None else 0        

    @property
    def receiver_nodes(self):
        if not hasattr(self, "receiver_nodes"):
            # reference: https://discuss.pytorch.org/t/batched-index-select/9115/6
            dummy = self.edge_rs[:,:,0].unsqueeze(-1)
            dummy = dummy.expand(self.batch_size, self.n_edges, self.node_dim)
            self.receiver_nodes = self.nodes.gather(1, dummy)
        return self.receiver_nodes

    @property
    def sender_nodes(self):
        if not  hasattr(self, "sender_nodes"):
            dummy = self.edge_rs[:,:,1].unsqueeze(-1)
            dummy = dummy.expand(self.batch_size, self.n_edges, self.node_dim)
            self.sender_nodes = self.nodes.gather(1, dummy)
        return self.sender_nodes

    def expanded_global_u(self, size):
        if self.global_u is None:
            return None 
        return self.global_u.unsqueeze(1).expand(self.batch_size, size, self.u_dim)

    def adj_sender_edge_index(self):
        """ return list of list of list indexing adjacent edge for each node (as receiver)
        """
        sender_index = [
            [[] for _ in range(self.n_nodes)] 
            for _ in range(self.batch_size)
        ]
        for b in range(self.batch_size):
            for i in range(self.n_edges):
                # edge is placeholder
                if int(self.edge_masks[b,i]) == 0:
                    continue 
                r, s = self.edge_rs[b,i,0], self.edge_rs[b,i,1]
                sender_index[b][r].append(i)
        return sender_index


class NNUpdate(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, nonlin=F.relu):
        super(NNEdgeUpdate, self).__init__()
        self.nonlin = nonlin
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        out = self.nonlin(self.fc1(x))
        out = self.nonlin(self.fc2(out))
        out = self.fc3(out)
        return out 


class NNMerge(nn.Module):
    def __init__(self):
        super(NNMerge, self).__init__()

    def forward(self, x):
        """ (k,D) -> (D,) """
        return torch.sum(x, -2)


class GNBlock(nn.Module):
    """ Full GN Block, with all 3 update and aggregation functions
    reference: https://arxiv.org/pdf/1806.01261.pdf
    reference2: https://github.com/deepmind/graph_nets/tree/master/graph_nets
    """
    def __init__(self, 
        edge_update, node_update, global_update,
        e2v_merge, e2u_merge, v2u_merge
    ):
        """ Inputs: (parameterizable functions/nn.Modules) """
        super(GNBlock, self).__init__()
        self.edge_update = edge_update
        self.node_update = node_update
        self.global_update = global_update
        self.e2v_merge = e2v_merge
        self.e2u_merge = e2u_merge
        self.v2u_merge = v2u_merge

    def edge_block(self, graphs):
        """ run edge update step 
        """
        edge_in = torch.cat([
            graphs.edges, 
            graphs.receiver_nodes,
            graphs.sender_nodes,
            graphs.expanded_global_u(graphs.n_edges)
        ], -1)
        edges_p = self.edge_update(edge_in)   
        return edges_p


    def node_block(self, graphs, edges_p=None):
        """ run node update step
            edges_p: (B,Ne,De')
        """
        sender_index = graphs.adj_sender_edge_index()   # (B,N,k) k variable length
        node_p_size = None

        edges_e2v = []
        for b in range(graphs.batch_size):
            temp = [None] * graphs.n_nodes

            for i in range(graphs.n_nodes):
                idx = sender_index[b][i]
                # no senders or just placeholder
                if len(idx) == 0:
                    continue 
                # get set of adjacency edges
                adj_edges = torch.stack([edges_p[b,k] for k in idx], 0) # (k,De)
                e2v_out = self.e2v_merge(adj_edges) # (De',)
                # dynamically infer e2v output size
                if node_p_size is None:
                    node_p_size = len(e2v_out)
                temp[i] = e2v_out

            # pack to tensor, (N,De')
            b_e2v = torch.stack([
                st if st is not None else torch.zeros(node_p_size) 
                for st in temp
            ], 0)   
            edges_e2v.append(b_e2v)   
        # pack to tensor, (B,N,De') 
        edges_e2v = torch.stack(edges_e2v, 0)
        
        # node update 
        node_in = torch.cat([
            edges_e2v,
            graphs.nodes,
            graphs.expanded_global_u(graphs.n_nodes)
        ], -1)
        nodes_p = self.node_update(node_in)
        return nodes_p


    def global_block(self, graphs, edges_p, nodes_p):
        """ run global node/attribute update step
            edges_p: (B,Ne,De')
            nodes_p: (B,Nv,Dv')
        """
        edges_e2u = self.e2u_merge(edges_p)    # (B,Du')
        nodes_v2u = self.v2u_merge(nodes_p)    # (B,Du'')
        global_in = torch.cat([
            edges_e2u,
            nodes_v2u,
            graphs.global_u
        ], -1)
        global_u_p = self.global_update(global_in)
        return global_u_p


    def forward(self, graphs):
        """ apply 3 updates and aggregations on batch of graphs 
            takes in GraphTuple and outputs updated GraphTuple
        """
        edges_p = self.edge_block(graphs)
        nodes_p = self.node_block(graphs, edges_p)
        global_u_p = self.global_block(graphs, edges_p, nodes_p)
        # update graphs
        out = GraphTuple(
            nodes_p, edges_p, global_u_p,
            edge_rs=graphs.edge_rs,
            node_masks=graphs.node_masks, 
            edge_masks=graphs.edge_masks,
        )
        return out 


        


