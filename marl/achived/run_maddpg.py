import os
import time
import argparse
import numpy as np
from pathlib import Path
from functools import partial 
from gym.spaces import Box, Discrete
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from collections import OrderedDict
from types import SimpleNamespace as SN


# local
from algorithms.maddpg import MADDPG
from runners.make_env import ENV_MAP
from runners.env_wrappers import SubprocVecEnv, DummyVecEnv
# from runenrs.buffer import ReplayBuffer
from runners.replay_buffer import EpisodeReplayBuffer
from runners.maddpg_runner import EpisodeRunner
from utils.exp_utils import setup_experiment, ExperimentLogger
from utils.exp_utils import time_left, time_str
from evaluate import maddpg_rollouts


#####################################################################################
### arguments 
#####################################################################################

def parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="maddpg",
                        help="name of the experiment")
    parser.add_argument("--save_dir", type=str, default="./exps",
                        help="top level path to save experiment/training results")
    parser.add_argument("--sub_dir", type=str, nargs='+',
                        help="sub folders for experiment (hierarchical), e.g. sub=a b c --> local-dir/a/b/c")
    parser.add_argument("--tag", type=str, nargs='+',
                        help="additional info for experiment, i.e. hyperparameters")
    parser.add_argument("--seed", default=1, type=int,
                        help="Random seed, if 0, do not set seed")
    parser.add_argument("--restore", type=str, default=None,
                        help="directory in which training state and model are loaded")
    ## NOTE: episode-wise or transition-wise (per transtion now, easier to log)
    parser.add_argument("--log_interval", default=1000, type=int,
                        help="frequency to log exploration/runner stats")
    parser.add_argument("--train_interval", default=0, type=int,
                        help="number of steps collected before each train")
    # parser.add_argument("--steps_per_update", default=100, type=int,
    #                     help="number of env steps collected before 1 training update")
    parser.add_argument("--target_update_interval", default=100, type=int,
                        help="syncing parameters with target networks")
    parser.add_argument("--train_log_interval", default=1000, type=int,
                        help="frequency to log training stats, e.g. losses")
    parser.add_argument("--eval_interval", default=1000, type=int,
                        help="number of steps collected before each evaluation")
    parser.add_argument("--save_interval", default=100000, type=int)
    
    # misc 
    parser.add_argument("--cuda", default=False, action='store_true')
    parser.add_argument("--cluster", default=False, action='store_true', 
                        help='if running in cluster (allow more resources)')
    parser.add_argument("--overwrite", type=str, nargs='+',
                        help="overwrite env config with format: nested_name nested_type value ...")
    parser.add_argument("--use_tensorboard", default=False, action='store_true',
                        help="if to use tensorboard for logging")
    parser.add_argument("--show_visual_range", default=False, action='store_true', 
                        help='if to show agent visual range when rendering')
    
    # Environment
    parser.add_argument("--env", type=str, default="mpe-hier",
                        help="name of the environment", choices=["mpe", "mpe_hier"])
    parser.add_argument("--scenario", type=str, default="simple_spread",
                        help="name of the scenario script")
    parser.add_argument("--env_config", type=str, default="",
                        help="file to environment scenario config")
    ## max episode length for termination
    parser.add_argument("--episode_length", default=25, type=int,
                        help="max episode length")
    parser.add_argument("--agent_alg", default="MADDPG", type=str,
                        help="agent model type", choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg", default="MADDPG", type=str,
                        help="adversary model type", choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action", action='store_true')

    # training 
    parser.add_argument("--n_episodes", default=20000, type=int,
                        help="max number of episodes to sample")
    ## for non-early-terminated episodes, n_env_steps ~= n_episodes * episode_length
    parser.add_argument("--n_env_steps", default=500000, type=int,
                        help="max number of env steps to sample")
    ## NOTE: episode-wise or step-wise (episode now)
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for model training per update")
    ## in case train batch size too large, could use smaller batch size 
    ## but multiple rounds of updates
    parser.add_argument("--n_updates_per_train", default=1, type=int,
                        help="number of updates per training round")
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="discount factor")
    
    # exploration/sampling 
    ## NOTE: episode-wise or transition-wise (per episodes now)
    parser.add_argument("--sample_batch_size", default=8, type=int,
                        help="number of data points sampled () per run")
    parser.add_argument("--max_buffer_size", default=40000, type=int,
                        help="maximum number of samples (episodes) to save in replay buffer")
    # parser.add_argument("--max_buffer_size", default=int(1e6), type=int,
    #                     help="maximum number of samples (transitions) to save in replay buffer")
    parser.add_argument("--n_exploration_eps", default=25000, type=int,
                        help="what is this ???")
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--n_step", type=int, default=1,
                        help="length of multistep value backup")

    # model 
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--critic", type=str, default="mlp",
                        help="type of critic network", choices=["mlp", "recurrent", "graph"])
    parser.add_argument("--actor", type=str, default="mlp",
                        help="type of actor network", choices=["mlp", "recurrent", "graph"])

    # evaluation 
    parser.add_argument("--no_eval", default=False, action='store_true',
                        help="do evaluation during training")
    parser.add_argument("--eval_n_episodes", default=20, type=int)

    # parallelism 
    parser.add_argument("--n_rollout_threads", default=4, type=int, 
                        help="number of parallel sampling workers to use")
    parser.add_argument("--n_training_threads", default=4, type=int)

    args = parser.parse_args()
    return args 


#####################################################################################
### funcs
#####################################################################################

def get_sample_scheme(n_agents, obs_dims, act_dims):
    """ get sample batch and buffer specifications 
    """
    scheme = OrderedDict()
    for i in range(n_agents):
        obs_dim, act_dim = obs_dims[i], act_dims[i]
        agent_scheme = {
            "obs": {"vshape": (obs_dim,)},
            "action": {"vshape": (act_dim,)}}
            "reward": {"vshape": (1,)},
            "next_obs": {"vshape": (obs_dim.,)},
            "done": {"vshape": (1,), "dtype": torch.uint8},
        }
        tag = "_{}".format(i)   
        agent_scheme = {k+tag: v for k, v in agent_scheme.items()}
        scheme.update(agent_scheme)
    return scheme
    
    

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


#####################################################################################
### main
####################################################################################

def run(args):
    """ main entry func """
    config = setup_experiment(args)
    logger = ExperimentLogger(config.save_dir, log_std_out=True, use_tensorboard=config.use_tensorboard)

    # make sampling runner  
    if not config.cuda:
        torch.set_num_threads(config.n_training_threads)
    env_func = ENV_MAP[config.env]
    p_env_func = partial(env_func, config.scenario, benchmark=False, 
                        show_visual_range=config.show_visual_range)
    env = make_parallel_env(p_env_func, config.env_config, config.n_rollout_threads, config.seed)
    if not config.no_eval:
        eval_env = env_func(config.scenario, benchmark=False, 
                        show_visual_range=config.show_visual_range, **config.env_config)

    # make learner agent 
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    replay_buffer = ReplayBuffer(config.max_buffer_size, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    
    # train loop 
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        logger.info("Episodes (%i-%i)/%i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device=config.device)

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            # list of (N) obs, each obs is (B,D) 
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]

            # get actions as torch Variables, list of (N) actions, each (B,D)
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]

            # step env and collect transition 
            next_obs, rewards, dones, infos = env.step(actions)
            # note: obs is (B,N,D), agent_actions is (N,B,D)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs

            t += config.n_rollout_threads   # total number of steps/transitions
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):  # coz t increment by 4
                # 1 update step
                maddpg.prep_training(device=config.device)
                for u_i in range(config.n_rollout_threads): # as if each process does optimization in parallel
                    for a_i in range(maddpg.nagents):
                        # each agent can have different collective experience samples
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=config.cuda)
                        maddpg.update(sample, a_i, logger=logger)
                    # sync target networks 
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device=config.device)

        # after eps_length * #threads steps are collected
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        # checkpoint 
        # if (epi > 0) and (ep_i % config.save_interval < config.n_rollout_threads):  # coz ep_i increment by 4
        if (epi > 0) and (ep_i % config.save_interval < config.n_rollout_threads): 
            os.makedirs(config.save_dir + "/checkpoints", exist_ok=True)
            maddpg.save(config.save_dir + "/checkpoints" + "/model_ep%i.ckpt" % (ep_i + 1))
            maddpg.save(config.save_dir + "/model.ckpt")

        # evaluation 
        if (not config.no_eval) and (epi > 0) and 
            (ep_i % config.eval_interval < config.n_rollout_threads): 
            rollouts = maddpg_rollouts(maddpg, eval_env, n_episodes=config.eval_n_episodes, 
                                    episode_length=config.episode_length, logger=logger, 
                                    render=False, save_gifs=True, fps=20) 


    # clean up 
    maddpg.save(config.save_dir + "/model.ckpt")    # final save
    env.close()
    logger.export_scalars_to_json("summary.json")
    logger.close()



if __name__ == '__main__':
    args = parse_args()
    run(args)