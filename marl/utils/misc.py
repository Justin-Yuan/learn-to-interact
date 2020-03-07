import os
import gym
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


def discount(x, gamma, axis=0):
    """ discount on rewards/values sequence, axis is time dimension in x
        reference: https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation
        reference2: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html 
    """
    y = scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=axis)
    return y[::-1]


def compute_advantages(
        last_r,
        rewards,
        vf_preds,
        gamma=0.9,
        lambda_=1.0,
        use_gae=True,
        use_critic=True,
        norm_advantages=False
    ):
    """ normal advantage or generalized advantage estiamate 
        reference: https://arxiv.org/pdf/1506.02438.pdf
        reference2: https://ray.readthedocs.io/en/latest/_modules/ray/rllib/evaluation/postprocessing.html
    - TODO: need to add time limit and termination masks (following) 
        reference3: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py
    Arguments:
        - last_r: (B,1)
        - rewards: (B,T,1)
        - vf_preds: (B,T,1)
    Returns:
        - advantages: (B,T,1)
        - value_targets: (B,T,1)
    """
    if use_gae:
        # use generalized advantage estimator
        vpred_t = torch.cat([vf_preds.squeeze(), last_r], -1) # (B,T+1)
        delta_t = rewards.squeeze() + gamma * vpred_t[:,1:] - vpred_t[:,:-1] # (B,T)
        
        advantages = discount(delta_t.numpy(), gamma * lambda_, -1) # (B,T)
        advantages = torch.from_numpy(advantages).unsqueeze(-1) # (B,T,1)
        value_targets = advantages + vf_preds
    else:
        # use rewards + value prediction (last step)
        rewards_plus_v = torch.cat([rewards.squeeze(), last_r], -1) # (B,T+1)
        discounted_returns = discount(rewards_plus_v.numpy(), gamma, -1)[:-1] # (B,T)
        discounted_returns = torch.from_numpy(discounted_returns).unsqueeze(-1) # (B,T,1)

        if use_critic:
            advantages = discounted_returns - vf_preds  # (B,T,1)
            value_targets = discounted_returns
        else:
            advantages = discounted_returns
            value_targets = torch.zeros_like(advantages)

    # if to normalize advantages
    if norm_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    return advantages, value_targets



def explained_variance_torch(y, pred):
    y_var = torch.pow(y.std(0), 0.5)
    diff_var = torch.pow((y - pred).std(0), 0.5)
    var = max(-1.0, 1.0 - (diff_var / y_var).data)
    return var