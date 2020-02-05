import numpy as np 
from copy import deepcopy
from functools import partial 
from collections import defaultdict
import torch 
from torch.autograd import Variable

from runners.base_runner import BaseRunner
from runners.sample_batch import EpisodeBatch 
from agents import DEFAULT_ACTION


#####################################################################################
### run an entire episode each time 
#####################################################################################

class EpisodeRunner(BaseRunner):
    """ wrap upon vectorized env to collect episdoes (vec env collect steps)
    """
    def __init__(self, scheme, env, mac, logger, batch_size, max_episode_len, 
                device="cpu", t_env=0, is_training=True, **kwargs):
        """ Arguments: 
            - scheme: sample batch specs
            - env: vectorized (parallelized) env 
            - mac: multiagent controller (e.g. maddpgs)
            - t_env: total env step so far using runner, used when restoring training
        """
        assert batch_size == env.nenvs, "batch_size should equal number of envs in vec_env"
        self.scheme = scheme
        self.env = env
        self.mac = mac
        self.logger = logger 
        self.batch_size = batch_size
        self.max_episode_len = max_episode_len
        self.device = device
        self.t_env = t_env  
        self.is_training = is_training
        
        # batch builder stuff
        self.new_batch = partial(EpisodeBatch, self.scheme, self.batch_size, 
                                self.max_episode_len, device=device)
        self.t = 0  # current step in episode 

        # summaries / metrics
        self.returns = []
        self.agent_returns = defaultdict(list)
        self.stats = {}


    def batch_builder(self, render=False):
        """ options to use different scheme for batch building, 
        e.g. rendering, entropy record
        """
        if render:
            # register new scheme if doesn't exist yet 
            if not getattr(self, "has_render_scheme", False):
                self.has_render_scheme = True 
                # get frame size dynamically & update scheme (frame is not per agent)
                _ = self.env.reset() 
                frame = self.env.get_images()[0]
                h, w, c = frame.shape
                self.height = h 
                self.width = w 
                self.channel = c 
                self.render_scheme = deepcopy(self.scheme)
                self.render_scheme["frame"] = {"vshape": (h,w,c)}
                self.new_render_batch = partial(EpisodeBatch, self.render_scheme, self.batch_size, 
                                    self.max_episode_len, device=self.device)
            return self.new_render_batch()
        else:
            return self.new_batch()


    def run(self, render=False):
        """ get batch of episodes
        Returns:
            batch:  EpisodeBatch
            results: dict of current episodes summaries
        """
        episode_lengths = [0 for _ in range(self.batch_size)]
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_agent_returns = {
            i: [0 for _ in range(self.batch_size)] 
            for i in range(self.mac.nagents)
        }
        # empty container: EpisodeBatch
        batch = self.batch_builder(render=render)
        results = {}

        # # NOTE: not used (unless env can terminate early, 
        # # consider moving paralleliation logic here too)
        # terminated = [False for _ in range(self.batch_size)]
        # envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]

        self.t = 0
        self.mac.init_hidden(self.batch_size)
        obs = self.env.reset()  # (B,N,O) or dict of (B,N,O)
        # # frames length is max_length + 1 
        # if render:
        #     frames = self.env.get_images()

        for et_i in range(self.max_episode_len):
            # (B,N,D) -> [(B,D)]*N or [ [dict (D,)]*N ]*B -> [dict (B,D)]*N
            torch_obs = self.dispatch_observations(obs)

            # [dict (B,A)]*N -> [ [(A,)]*N ]*B or [ [tuple (A,)]*N ]*B
            torch_actions = self.mac.step(torch_obs, explore=self.is_training)
            actions = self.group_actions(torch_actions)

            # step env and collect transition, each is (B,N,D)
            next_obs, rewards, dones, infos = self.env.step(actions)
            frames = None if not render else self.env.get_images() # (B,H,W,C)
            # if render:
            #     self.env.render()

            # incrementally build episode batch 
            transition_data = self.pack_transition(
                obs, torch_actions, next_obs, rewards, dones, frames=frames)
            batch.update(transition_data, ts=self.t)

            # update episode stats 
            for i in range(self.env.nenvs):
                episode_lengths[i] += 1
                episode_returns[i] += sum(rewards[i])
                for a in range(self.mac.nagents):
                    episode_agent_returns[a][i] += rewards[i, a] 

            # Move onto the next timestep
            obs = next_obs
            self.t += 1
            if self.is_training:   # only update counter when training 
                self.t_env += self.env.nenvs

        # NOTE: summaries 
        results["n_episodes"] = self.batch_size 
        results["ep_lengths"] = deepcopy(episode_lengths)
        self.stats["n_episodes"] = self.batch_size + self.stats.get("n_episodes", 0)
        self.stats["ep_lengths"] = sum(episode_lengths) + self.stats.get("ep_length", 0)
        
        ### episode returns (sum of agent returns)
        results["returns"] = deepcopy(episode_returns)
        self.returns.extend(episode_returns)

        ### per-agent episode returns
        for i in range(self.mac.nagents):
            agent_returns = episode_agent_returns[i]
            results["agent_{}_returns".format(i)] = deepcopy(agent_returns)
            self.agent_returns[i].extend(agent_returns)

        return batch, results

