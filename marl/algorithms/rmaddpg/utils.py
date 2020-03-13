import os
import time
import random 
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
        sample: SampleBatch/EpisodeBatch, each is (B,T,D)
        scheme: multi-agent sample scheme 
    Returns:
        obs, acs, rews, next_obs, dones: each is [(B,T,D)]*N
        obs, next_obs, action can be [dict (B,T,D)]*N
    """
    def filter_key(key):
        if ("obs" in key) and (not "next_obs" in key):  # for obs
            return [k for k in scheme if key in k and not "next_obs" in k]
        else:
            return [k for k in scheme if key in k]  

    if fields is None:
        fields = ["obs", "action", "reward", "next_obs", "done"]    # default 
    # each should be [(B,T,D)]*N or [dict (B,T,D)]*N
    parsed = [[] for _ in range(len(fields))]

    # import pdb; pdb.set_trace()
    for f_i, f in enumerate(fields):
        for i in range(n_agents):
            matched_keys = filter_key("{}/{}".format(f, i))
            if len(matched_keys) > 1:   # dict for sub_fields
                field = {
                    key.split("/")[-1]: sample[key]
                    for key in matched_keys
                } # dict (B,T,D) 
            else:
                field = sample[matched_keys[0]] # (B,T,D) 
            parsed[f_i].append(field) 
    return parsed 
     

def make_parallel_env(env_func, env_config, batch_size, n_rollout_threads, seed):
    # func wrapper with seed (for training)
    def get_env_fn(rank):
        def init_env():
            # do not set seed i if -1 (e.g. for evaluation)
            if seed >= 0:
                # env.seed(seed + rank * 1000)
                random.seed(seed + rank * 1000)
                np.random.seed(seed + rank * 1000)
                torch.manual_seed(seed + rank * 1000)
                # mpe has its own seeding 
                env = env_func(seed=seed + rank * 1000, **env_config)
            else:
                env = env_func(**env_config)
            return env
        return init_env

    envs = [get_env_fn(i) for i in range(batch_size)]
    if n_rollout_threads > 1:
        return SubprocVecEnv(envs, n_workers=n_rollout_threads)
    else: 
        # can use in evaluation (with seed -1)
        return DummyVecEnv(envs)


def log_results(t_env, results, logger, mode="sample", episodes=None, 
        log_video=True, display_eps_num=4, log_agent_returns=False, **kwargs):
    """ training & evaluation logging 
    Arguments:
        - t_env: env step 
        - results: result dicts
        - logger: experiment logger
        - mode: sample|train|eval
    """
    # print current directory for easier identification 
    logger.info(logger.log_dir)

    if (mode == "sample") or (mode == "eval"):
        # exploration/evaluation episode stats, e.g. returns, lengths
        returns = results["returns"]
        agent_returns = results["agent_returns"]

        logger.add_scalar("{}/returns_mean".format(mode), np.mean(returns), t_env)
        logger.add_scalar("{}/returns_std".format(mode), np.std(returns), t_env)
        if log_agent_returns:
            for k, a_returns in agent_returns.items():
                logger.add_scalar("{}/{}_returns_mean".format(mode, k), np.mean(a_returns), t_env)
                logger.add_scalar("{}/{}_returns_std".format(mode, k), np.std(a_returns), t_env)

        # log videos 
        if log_video and episodes is not None and "frame" in episodes.scheme:
            frames = episodes["frame"]  # (B,T,H,W,C)
            b, t, h, w, c = frames.shape
            display_num = min(b, display_eps_num) 
            frames = frames[:display_num] 
            # # tb accepts (N,T,C,H,W)
            # vid_tensor = frames.permute(0,1,4,2,3) * 255
            # logger.add_video("{}/frames".format(mode), vid_tensor, t_env)
            # save to local 
            stacked_frames = frames.data.cpu().numpy().astype(np.uint8).reshape(-1,h,w,c)
            logger.log_video("{}_video.gif".format(mode), stacked_frames)

        log_str = "t_env: {} | mean returns: {:.2f}".format(t_env, np.mean(returns))
        # temp = ", ".join(["agent_{}: {:.2f}".format(k, np.mean(v)) for k, v in sorted(agent_returns.items())])
        # log_str += " | " + temp
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

        log_str = "t_env: {}".format(t_env)
        temp = " | ".join(["{}: {:.2f}".format(k, np.mean(v)) for k, v in sorted(loss_dict.items())])
        log_str += " | " + temp
        logger.info(log_str)
        
    else:
        raise NotImplementedError("logging option not supported!")



def log_weights(learner, logger, t_env):
    """ log network weights for debug 
    """
    for i, agent in enumerate(learner.agents):
        agent_actor_params = {
            "policy_{}_{}".format(i, k): v for k, v in agent.policy.named_parameters()}
        agent_critic_params = {
            "critic_{}_{}".format(i, k): v for k, v in agent.critic.named_parameters()}
        logger.add_histogram_dict(agent_actor_params, t_env)
        logger.add_histogram_dict(agent_critic_params, t_env)