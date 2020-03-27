import numpy as np 
from copy import deepcopy
from collections import defaultdict 
import torch 
from torch.autograd import Variable
from agents import DEFAULT_ACTION



class BaseRunner(object):
    """ defines interface for Runner class 
        Runner bridges between env and trainer 
    """
    def __init__(self, scheme, env, mac, logger, **kwargs):
        self.scheme = scheme
        self.env = env
        self.mac = mac
        self.logger = logger

        # summaries / metrics
        self.returns = []
        self.agent_returns = defaultdict(list)
        self.stats = {}


    def get_summaries(self):
        """ get metrics for episode(s) run summaries """
        return {
            "returns": deepcopy(self.returns), 
            "agent_returns": deepcopy(self.agent_returns), 
            "stats": deepcopy(self.stats)
        }


    def reset_summaries(self):
        """ clear history after each runner log step """
        self.returns.clear()
        self.agent_returns.clear()

    
    def dispatch_observations(self, obs, to_torch=True):
        """ rearrange observations to be per agent, convert to Tensor
            (B,N,D) -> [(B,D)]*N or 
            [ [(D,)]*N ]*B -> [(B,D)]*N
            [ [dict (D,)]*N ]*B -> [dict (B,D)]*N
        """
        def collate_fn(x, to_torch=True):
            if not to_torch:
                return np.array(x)
            else:
                return Variable(torch.Tensor(x), requires_grad=False)

        if isinstance(obs, np.ndarray):
            # obs same shape across agents, obs stacked 
            torch_obs = [
                collate_fn(obs[:, i], to_torch=to_torch) for i in range(self.mac.nagents)
            ]   # (B,N,D) -> [(B,D)]*N

        elif all([isinstance(ob, (list, tuple)) for ob in obs]):
            # each sample obs is np array (different size across agents)
            torch_obs = [
                collate_fn(a_obs, to_torch=to_torch) for a_obs in zip(*obs)
            ]  # [ [(D,)]*B ]*N -> [(B,D)]*N

        elif all([isinstance(ob, dict) for ob in obs]):
            # each sample obs is dict 
            # each dict key to list of obs subfield arrays
            torch_obs = [defaultdict(list)] * self.mac.nagents  
            for i, b_dicts in enumerate(zip(*obs)): # [ [dict (D,)]*B ]*N
                for b_dict in b_dicts:
                    for k in b_dict:
                        torch_obs[i][k].append(b_dict[k])
            # concat subfield arrays 
            torch_obs = [
                {
                    k: collate_fn(obs_list, to_torch=to_torch)
                    for k, obs_list in a_dict.items()
                } for a_dict in torch_obs
            ]   # [ [dict (D,)]*N ]*B -> [dict (B,D)]*N
        else:
            # cannot dispatch obs 
            raise Exception("Unrecognized agent observation format...")
        return torch_obs


    def group_actions(self, actions):
        """ rearrange actions to be per environment
            [dict (B,A)]*N -> [ [(A,)]*N ]*B or [ [dict (A,)]*N ]*B
        """
        # # rearrange actions to be per environment, (N,B,A) -> (B,N,A)
        # actions = [[ac[i] for ac in agent_actions] for i in range(self.env.nenvs)]
        env_actions = [[] for _ in range(self.batch_size)]
        for i, action in enumerate(actions):    # dict (B,A)
            if DEFAULT_ACTION in action:
                for b in range(self.batch_size):
                    act = action[DEFAULT_ACTION][b].data.numpy()
                    env_actions[b].append(act)  # list at b is [(A,)]*N 
            else:
                for b in range(self.batch_size):
                    act = {k: ac[b].data.numpy() for k, ac in action.items()}
                    env_actions[b].append(act)  # list at b is [dict (A,)]*N
        return env_actions


    def pack_transition(self, obs, actions, next_obs, rewards, dones, frames=None, **kwargs):
        """ regroup transitions for batch builder or replay buffer
        transition_data is {obs_1: , action_1:, ..., obs_n:, action_n:, ...}
        Arguments:
            obs, next_obs: (B,N,D) or [ [(D,)]*N ]*B or [ [dict (D,)]*N ]*B
            actions: [dict (B,A)]*N
            rewards, dones: (B,N,1)
            frame: (B,H,W,C)
        Returns:
            transition: dict of all key-batch_values for buffer
        """
        transition = {}
        # obs & next_obs 
        if isinstance(obs, dict) and isinstance(next_obs, dict):
            # [dict (B,D)]*N
            dp_obs = self.dispatch_observations(obs, to_torch=False) 
            dp_next_obs = self.dispatch_observations(next_obs, to_torch=False)

            for i in range(self.mac.nagents):
                for k, obs_k in obs.items():
                    transition["obs/{}/{}".format(i,k)] = dp_obs[i][k]  # (B,D)
                    transition["next_obs/{}/{}".format(i,k)] = dp_next_obs[i][k]
                    # transition["obs/{}/{}".format(i,k)] = obs[k][:, i]  # (B,D)
                    # transition["next_obs/{}/{}".format(i,k)] = next_obs[k][:, i]

        elif isinstance(obs, (list, tuple)) and isinstance(next_obs, (list, tuple)):
            # [(B,D)]*N
            dp_obs = self.dispatch_observations(obs, to_torch=False) 
            dp_next_obs = self.dispatch_observations(next_obs, to_torch=False)

            for i in range(self.mac.nagents):
                transition["obs/{}".format(i)] = dp_obs[i]  # (B,D)
                transition["next_obs/{}".format(i)] = dp_next_obs[i]

        else:   # (B,N,D)
            for i in range(self.mac.nagents):
                transition["obs/{}".format(i)] = obs[:, i]  # (B,D)
                transition["next_obs/{}".format(i)] = next_obs[:, i]
        
        # actions
        for i, action in enumerate(actions): # action is dict (B,A)
            if DEFAULT_ACTION in action:
                act = torch.stack([
                    action[DEFAULT_ACTION][b]
                    for b in range(self.batch_size)
                ], 0)   # (B,A)
                transition["action/{}".format(i)] = act
            else:
                for k in action:
                    act = torch.stack([
                        action[k][b]
                        for b in range(self.batch_size)
                    ], 0)   # (B,A)
                    transition["action/{}/{}".format(i,k)] = act

        # rewards, dones 
        for i in range(self.mac.nagents):
            transition["reward/{}".format(i)] = rewards[:, i]
            transition["done/{}".format(i)] = dones[:, i]

        # others 
        if frames is not None:
            transition["frame"] = frames
        # hack for flexibility 
        for k, v in kwargs:
            assert k in self.scheme:
            transition[k] = v
        return transition


    def batch_builder(self):
        pass 

    def run(self, render=False):
        pass 

