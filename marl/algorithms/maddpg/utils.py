import os
import time
import numpy as np
from gym.spaces import Box, Discrete, Dict
from collections import OrderedDict, defaultdict
import torch

# local
from runners.env_wrappers import SubprocVecEnv, DummyVecEnv

#####################################################################################
### funcs
#####################################################################################

def get_sample_scheme(n_agents, obs_spaces, act_spaces):
    """ get sample batch and buffer specifications 
        specify how to encode experience in SampleBatch/EpisodeBatch
        use hierarchical keys: obs/agent_idx/obs_sub_fields
    """
    scheme = OrderedDict()
    for i in range(n_agents):
        obs_space, act_space = obs_spaces[i], act_spaces[i]
        # observation space(s)
        if isinstance(obs_space, Dict):
            for k, sp in obs_space.spaces.items():
                obs_dim = sp.shape[0] if isinstance(sp, Box) else sp.n 
                scheme["obs/{}/{}".format(i,k)] = {"vshape": (obs_dim,)}
                scheme["next_ob/{}/{}".format(i,k)] = {"vshape": (obs_dim,)}
        else:
            obs_dim = obs_space.shape[0] if isinstance(obs_space, Box) else obs_space.n 
            scheme["obs/{}".format(i)] = {"vshape": (obs_dim,)}
            scheme["next_obs/{}".format(i)] = {"vshape": (obs_dim,)}

        # action space(s)
        if isinstance(act_space, Dict):
            for k, sp in act_space.spaces.items():
                act_dim = sp.shape[0] if isinstance(sp, Box) else sp.n 
                scheme["action/{}/{}".format(i,k)] = {"vshape": (act_dim,)}
        else:
            act_dim = act_space.shape[0] if isinstance(act_space, Box) else act_space.n 
            scheme["action/{}".format(i)] = {"vshape": (act_dim,)}

        # others 
        scheme["reward/{}".format(i)] = {"vshape": (1,)}
        scheme["done/{}".format(i)] = {"vshape": (1,), "dtype": torch.uint8}
    return scheme


def dispatch_samples(sample, scheme, n_agents, fields=None):
    """ transform samples from buffer to feed in maddpg learner 
        specify how to decode sample to per-agent experience 
    Arguments:
        sample: SampleBatch/EpisodeBatch, each is (B,D)
        scheme: multi-agent sample scheme 
    Returns:
        obs, acs, rews, next_obs, dones: each is [(B,D)]*N
        obs, next_obs, action can be [dict (B,D)]*N
    """
    def filter_key(key):
        if ("obs" in key) and (not "next_obs" in key):  # for obs
            return [k for k in scheme if key in k and not "next_obs" in k]
        else:
            return [k for k in scheme if key in k]  

    if fields is None:
        fields = ["obs", "action", "reward", "next_obs", "done"]    # default 
    # each should be [(B,D)]*N or [dict (B,D)]*N
    parsed = [[] for _ in range(len(fields))]

    # import pdb; pdb.set_trace()
    for f_i, f in enumerate(fields):
        for i in range(n_agents):
            matched_keys = filter_key("{}/{}".format(f, i))
            if len(matched_keys) > 1:   # dict for sub_fields
                field = {
                    key.split("/")[-1]: sample[key]
                    for key in matched_keys
                } # dict (B,D) 
            else:
                field = sample[matched_keys[0]] # (B,D) 
            parsed[f_i].append(field) 
    return parsed 
     

def make_parallel_env(env_func, env_config, batch_size, n_rollout_threads, seed):
    # func wrapper with seed (for training)
    def get_env_fn(rank):
        def init_env():
            env = env_func(**env_config)
            # do not set seed i if -1 (e.g. for evaluation)
            if seed >= 0:
                env.seed(seed + rank * 1000)
                random.seed(seed + rank * 1000)
                np.random.seed(seed + rank * 1000)
            return env
        return init_env

    envs = [get_env_fn(i) for i in range(batch_size)]
    if n_rollout_threads > 1:
        return SubprocVecEnv(envs, n_workers=n_rollout_threads)
    else: 
        # can use in evaluation (with seed -1)
        return DummyVecEnv(envs)


def log_results(t_env, results, logger, mode="sample"):
    """ training & evaluation logging 
    Arguments:
        - t_env: env step 
        - results: result dicts
        - logger: experiment logger
        - mode: sample|train|eval
    """
    if (mode == "sample") or (mode == "eval"):
        # exploration/evaluation episode stats, e.g. returns, lengths
        returns = results["returns"]
        agent_returns = results["agent_returns"]

        logger.add_scalar("{}/returns_mean".format(mode), np.mean(returns), t_env)
        logger.add_scalar("{}/returns_std".format(mode), np.std(returns), t_env)
        for k, a_returns in agent_returns:
            logger.add_scalar("{}/{}_returns_mean".format(mode, k), np.mean(a_returns), t_env)
            logger.add_scalar("{}/{}_returns_std".format(mode, k), np.std(a_returns), t_env)

        log_str = "t_env: {} | mean returns: {}".format(t_env, np.mean(returns))
        temp = ", ".join(["{}: {}".format(k, np.mean(v)) for k, v in sorted(agent_returns.items())])
        log_str += " | " + temp
        logger.info(log_str)

    elif mode == "train":
        # training stats, e.g. losses
        agent_losses = results["agent_losses"]

        # group keys by agents and loss types
        agent_keys = defaultdict(list)
        loss_keys = defaultdict(list)
        for k in agent_losses.keys():   # e.g. agent_i/policy_loss
            tmp = k.split()
            agent_name, loss_name = tmp[0], tmp[-1]
            agent_keys[agent_name].append(k)
            loss_keys[loss_name].append(k)
        
        loss_dict = {}
        for loss_name, keys in loss_keys.items():
            # [ [loss] * #agents ] * #updates
            loss = list(zip(*[agent_losses[k] for k in keys]))
            loss = [np.sum(l) for l in loss]
            loss_dict[loss_name] = loss

            logger.add_scalar("{}/{}_mean".format(mode, loss_name), np.mean(loss), t_env)      
            logger.add_scalar("{}/{}_std".format(mode, loss_name), np.std(loss), t_env)      
    
        log_str = "t_env: {}".format(t_env）
        temp = " | ".join(["{}: {}".format(k, np.mean(v)) for k, v in sorted(loss_dict.items())])
        log_str += " | " + temp
        logger.info(log_str)
        
    else:
        raise NotImplementedError("logging option not supported!")
