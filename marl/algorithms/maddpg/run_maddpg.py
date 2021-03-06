import os
import sys 
# path at level marl/
sys.path.insert(0, os.path.abspath("."))
import time
import argparse
import numpy as np
from functools import partial 
from collections import OrderedDict, defaultdict
import torch

# local
from algorithms.maddpg.utils import get_sample_scheme, dispatch_samples
from algorithms.maddpg.utils import make_parallel_env, log_results
from algorithms.maddpg.utils import log_weights
from algorithms.maddpg.utils import switch_batch
from algorithms.maddpg import MADDPG

from runners.make_env import ENV_MAP
from runners.replay_buffer import ReplayBuffer
from runners.step_runner import StepRunner
from utils.exp_utils import setup_experiment, ExperimentLogger, ExperimentState
from utils.exp_utils import time_left, time_str, merge_dict


#####################################################################################
### arguments 
#####################################################################################

def parse_args():
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
    # if specified and not restore, will load model for experiment init
    # if also restore, will overwrite default path in restore_experiment
    parser.add_argument("--restore_model", type=str, default=None,
                        help="file in which model are loaded")
    ## NOTE: episode-wise or transition-wise (per transtion now, easier to log)
    parser.add_argument("--log_interval", default=1000, type=int,
                        help="frequency to log exploration/runner stats")
    parser.add_argument("--train_interval", default=100, type=int,
                        help="number of steps collected before each train")
    # parser.add_argument("--steps_per_update", default=100, type=int,
    #                     help="number of env steps collected before 1 training update")
    parser.add_argument("--target_update_interval", default=0, type=int,
                        help="syncing parameters with target networks")
    parser.add_argument("--train_log_interval", default=1000, type=int,
                        help="frequency to log training stats, e.g. losses")
    parser.add_argument("--eval_interval", default=10000, type=int,
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
    parser.add_argument("--env", type=str, default="mpe_hier",
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
    parser.add_argument("--sync_samples", default=False, action='store_true',
                        help="if to use synchronized samples for each agent training")
    parser.add_argument("--norm_rewards", default=False, action='store_true',
                        help="if to normalize rewards before training")
 
    
    # exploration/sampling 
    ## NOTE: episode-wise or transition-wise (per episodes now)
    parser.add_argument("--sample_batch_size", default=4, type=int,
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
    parser.add_argument("--norm_in", default=False, action='store_true',
                        help="if to normalize inputs to agent networks")
    parser.add_argument("--constrain_out", default=False, action='store_true',
                        help="if to use tanh for network final activation")
    # NOTE: moa stuff (optional)
    parser.add_argument("--model_of_agents", default=False, action='store_true',
                        help="if to use model of agents/approximate policies")
 
    # evaluation 
    parser.add_argument("--no_eval", default=False, action='store_true',
                        help="do evaluation during training")
    parser.add_argument("--no_render", default=False, action='store_true', 
                        help='if to stop rendering in evaluation rollouts')
    parser.add_argument("--eval_n_episodes", default=10, type=int)
    parser.add_argument("--eval_batch_size", default=2, type=int,
                        help="number of data points evaluated () per run")

    # loggings 
    parser.add_argument("--log_agent_returns", default=False, action='store_true',
                        help="if to log per agent returns on tensorboard")

    # parallelism 
    parser.add_argument("--n_rollout_threads", default=4, type=int, 
                        help="number of parallel sampling workers to use")
    parser.add_argument("--n_training_threads", default=4, type=int)

    args = parser.parse_args()
    return args 



#####################################################################################
### main
####################################################################################

def run(args):
    """ main entry func """
    # NOTE: experiment setup
    config, is_restore = setup_experiment(args)
    logger = ExperimentLogger(config.save_dir, log_std_out=True, use_tensorboard=config.use_tensorboard)
    if not config.cuda:
        torch.set_num_threads(config.n_training_threads)

    # NOTE: init/load experiment state
    estate = ExperimentState()
    if is_restore:
        estate.load_state(config.restore_exp_state)
    # make counter copies to reduce writing ...
    episode = estate.episode     # total episodes so far
    t_env = estate.t_env       # total env interacetion steps so far
    # t_max = config.n_env_steps    # max number of steps to runs
    t_max = config.n_episodes * config.episode_length 

    # NOTE: make vectorized env 
    env_func = ENV_MAP[config.env]
    p_env_func = partial(env_func, config.scenario, benchmark=False, 
                        show_visual_range=config.show_visual_range)
    env = make_parallel_env(p_env_func, config.env_config, config.sample_batch_size, 
                            config.n_rollout_threads, config.seed)
    if not config.no_eval:
        # NOTE: seed to eval env is differnet than env: seed + 100
        # also eval env always uses 1 rollout thread 
        eval_env = make_parallel_env(p_env_func, config.env_config, 
                                config.eval_batch_size, 1, config.seed+100)

    # NOTE: make learner agent 
    if is_restore or config.restore_model is not None:
        learner = MADDPG.init_from_save(config.restore_model)
    else:
        learner = MADDPG.init_from_env(
            env, 
            agent_alg=config.agent_alg,
            adversary_alg=config.adversary_alg,
            tau=config.tau,
            lr=config.lr,
            hidden_dim=config.hidden_dim,
            norm_in=config.norm_in,
            constrain_out=config.constrain_out,
            model_of_agents=config.model_of_agents
        )

    # NOTE: make sampling runner (env wrapper)  
    scheme = get_sample_scheme(learner.nagents, env.observation_space, env.action_space)
    runner = StepRunner(scheme, env, learner, logger, config.sample_batch_size,
                            config.episode_length, device=config.device, t_env=t_env,
                            is_training=True)
    if not config.no_eval:
        eval_runner = StepRunner(scheme, eval_env, learner, logger, 
                            config.eval_batch_size, config.episode_length, 
                            device=config.device, t_env=t_env, is_training=False)
    # NOTE: since in args, batch_size is episode-wise 
    train_batch_size_steps = config.batch_size * config.episode_length
    buffer = ReplayBuffer(scheme, config.max_buffer_size * config.episode_length, device=config.device,
                            prefill_num=1*train_batch_size_steps)

    # NOTE: start training
    logger.info("Beginning training")
    start_time = time.time()
    last_time = start_time
    
    ############################################
    # while t_env <= t_max:
    while episode <= config.n_episodes:

        learner.prep_rollouts(device=config.device)
        explr_pct_remaining = max(0, config.n_exploration_eps - episode) / config.n_exploration_eps
        learner.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        learner.reset_noise()

        # run an episode 
        for t_ep in range(config.episode_length):

            # NOTE: Run for one (parallel) step at a time
            batch = runner.run()
            buffer.insert_batch(batch)
            # update counters 
            t_env = runner.t_env
            estate.t_env = t_env

            ############################################
            # NOTE: training updates
            ## change to batch_size * n_updates_per_train for n_updates > 1
            if buffer.can_sample(train_batch_size_steps) and (t_env - estate.last_train_t >= config.train_interval):
                learner.prep_training(device=config.device)

                for _ in range(config.n_updates_per_train):

                    # NOTE: MOA updates (prior to actor and critic updates)
                    if config.model_of_agents:
                        # latest samples are the same across agents (no async)
                        moa_index = buffer.make_latest_index(train_batch_size_steps)
                        moa_sample = buffer.sample_index(moa_index)
                        moa_sample = dispatch_samples(moa_sample, scheme, learner.nagents)
                        for a_i in range(learner.nagents):
                            learner.update_moa(moa_sample, a_i)

                    # actor and critic updates
                    sample = None 
                    for a_i in range(learner.nagents):

                        if config.sync_samples:
                            # if not None, reuse episode_sample 
                            if sample is None:
                                sample = buffer.sample(train_batch_size_steps)
                        else:
                            # each agent can have different collective experience samples
                            sample = buffer.sample(train_batch_size_steps)

                        if sample.device != config.device:
                            sample.to(config.device)

                        # dispatch sample to per agent [(B,D)]*N 
                        sample = dispatch_samples(sample, scheme, learner.nagents)
                        learner.update(sample, a_i, norm_rewards=config.norm_rewards)  #, logger=logger)

                    # sync target networks 
                    if t_env - estate.last_target_update_t >= config.target_update_interval:
                        learner.update_all_targets()
                        estate.last_target_update_t = t_env

                learner.prep_rollouts(device=config.device)
                estate.last_train_t = t_env

                # collect & log trianing stats
                if t_env - estate.last_train_log_t >= config.train_log_interval:
                    train_results = learner.get_summaries()
                    learner.reset_summaries()
                    logger.info("\n")
                    logger.info("*** training log ***")
                    log_results(t_env, train_results, logger, mode="train")
                    estate.last_train_log_t = t_env

        ############################################
        # NOTE: logging (exploration/sampling)
        if (estate.last_log_t == 0) or (t_env - estate.last_log_t >= config.log_interval):
            logger.info("\n")
            logger.info("*** sampling log ***")
            # timing 
            logger.info("t_env: {} / {}, eps: {} / {}".format(
                t_env, t_max, episode, config.n_episodes))
            logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, estate.last_log_t, t_env, t_max), 
                time_str(time.time() - start_time)
            ))
            last_time = time.time()
            # log collected episode stats
            results = runner.get_summaries()
            runner.reset_summaries()
            log_results(t_env, results, logger, mode="sample", 
                        log_agent_returns=config.log_agent_returns)
            estate.last_log_t = t_env

        ############################################
        # NOTE: Execute test runs once in a while
        if not config.no_eval and ((estate.last_test_t == 0) or (t_env - estate.last_test_t >= config.eval_interval)):
            n_test_runs = max(1, config.eval_n_episodes // runner.batch_size)

            eval_episodes = [[] for _ in range(config.episode_length)]
            for _ in range(n_test_runs):
                for et in range(config.episode_length):
                    eval_bt = eval_runner.run(render=(not config.no_render))
                    eval_episodes[et].append(eval_bt)

            # collect evaluation stats
            eval_results = eval_runner.get_summaries()
            eval_runner.reset_summaries()
            # concat along batch dim -> list of concat' SampleBatch
            eval_episodes = [e[0].concat(e[1:]) for e in eval_episodes]
            # stack along time dim -> EpisodeBatch
            eval_episodes = eval_episodes[0].time_stack(eval_episodes[1:])

            logger.info("\n")
            logger.info("*** evaluation log ***")
            log_results(t_env, eval_results, logger, mode="eval", episodes=eval_episodes, 
                        log_agent_returns=config.log_agent_returns)
            estate.last_test_t = t_env

            # NOTE: debug noly, log network weights
            log_weights(learner, logger, t_env)

        ############################################
        # NOTE: checkpoint 
        if (estate.last_save_t == 0) or (t_env - estate.last_save_t >= config.save_interval): 
            os.makedirs(config.save_dir + "/checkpoints", exist_ok=True)
            learner.save(config.save_dir + "/checkpoints" + "/model_{}.ckpt".format(t_env))
            learner.save(config.save_dir + "/model.ckpt")
            logger.info("\n")
            logger.info("*** checkpoint log ***")
            logger.info("Saving models to {}".format(
                "/checkpoints" + "/model_{}.ckpt".format(t_env)
            ))
            estate.last_save_t = t_env
            estate.save_state(config.save_dir + "/exp_state.pkl")

        # end of current episode 
        episode += config.sample_batch_size
        estate.episode = episode
        
    ############################################
    # NOTE: clean up 
    learner.save(config.save_dir + "/model.ckpt")    # final save
    estate.last_save_t = t_env
    estate.save_state(config.save_dir + "/exp_state.pkl")

    env.close()
    logger.export_scalars_to_json("summary.json")
    logger.info("Finished Training")
    logger.close()
    


if __name__ == '__main__':
    args = parse_args()
    run(args)

