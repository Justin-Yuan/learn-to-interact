import os 
import sys 
# path at level marl/
sys.path.insert(0, os.path.abspath("."))
import time
import pickle 
import imageio
import argparse
import numpy as np
from functools import partial 
from pathlib import Path
import torch
from torch.autograd import Variable

from runners.make_env import ENV_MAP
from algorithms.rmaddpg import RMADDPG
from runners.sample_batch import EpisodeBatch
from runners.episode_runner import EpisodeRunner
from utils.exp_utils import setup_evaluation, ExperimentLogger 
from utils.exp_utils import time_left, time_str, merge_dict

from algorithms.rmaddpg.utils import get_sample_scheme, make_parallel_env, log_results


#####################################################################################
### arguments 
#####################################################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="maddpg_eval",
                        help="name of the experiment")
    parser.add_argument("--save_dir", type=str, default="./results",
                        help="top level path to save experiment/training results")
    parser.add_argument("--sub_dir", type=str, nargs='+',
                        help="sub folders for experiment (hierarchical), e.g. sub=a b c --> local-dir/a/b/c")
    parser.add_argument("--tag", type=str, nargs='+',
                        help="additional info for experiment, i.e. hyperparameters")
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed, if negative do not set seed")
    parser.add_argument("--cuda", default=False, action='store_true')

    # evaluation
    parser.add_argument("--restore", type=str, default=None,
                        help="directory in which training state and model are loaded")
    parser.add_argument("--restore_model", type=str, default=None,
                        help="file for the checkpoint file to load")
    parser.add_argument("--checkpoint", default=-1, type=int,
                        help="Load incremental policy from given episode (alternative to restore_model)")
    parser.add_argument("--copy_checkpoint", default=False, action="store_true",
                        help="if to copy the evaluated checkpoint to current eval directory")
    parser.add_argument("--n_episodes", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=2, type=int,
                        help="number of data points evaluated () per run")
    parser.add_argument("--save_gifs", default=False, action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--save_gifs_num", default=-1, type=int,
                        help="number of episode gifs to save")
    parser.add_argument("--fps", default=30, type=int, 
                        help="frame per second of generated gif")

    # Environment
    parser.add_argument("--env", type=str, default="mpe_hier",
                        help="name of the environment", choices=["mpe", "mpe_hier"])
    parser.add_argument("--scenario", type=str, default="simple_spread",
                        help="name of the scenario script")
    parser.add_argument("--env_config", type=str,
                        help="file to environment scenario config")
    parser.add_argument("--use_restore_env_config", default=False, action='store_true',
                        help="if to use env config from retore directory")
    parser.add_argument("--use_restore_config", default=False, action='store_true',
                        help="if to use exp config from retore directory")
    parser.add_argument("--episode_length", default=25, type=int,
                        help="max episode length")
    parser.add_argument("--show_visual_range", default=False, action='store_true', 
                        help='if to show agent visual range when rendering')
    
    # parallelism 
    parser.add_argument("--n_rollout_threads", default=1, type=int, 
                        help="number of parallel sampling workers to use")

    args = parser.parse_args()
    return args


#####################################################################################
### eval funcs 
####################################################################################

def maddpg_rollout_runner(config, learner, runner, n_episodes=10, episode_length=25, logger=None, 
                    render=False, save_dir=None, fps=20, save_gifs=False, save_gifs_num=-1, 
                    log_agent_returns=True, **kwargs):
    """ get rollouts with runner 
    """
    eval_episodes = []

    n_test_runs = max(1, n_episodes // runner.batch_size)
    assert n_episodes % runner.batch_size == 0, "n_episodes should be divisible by batch_size"
    learner.prep_rollouts(device=config.device)

    for _ in range(n_test_runs):
        eval_batch, eval_res = runner.run(render=render)
        eval_episodes.append(eval_batch)

    # collect evaluation stats
    eval_episodes = eval_episodes[0].concat(eval_episodes[1:])
    eval_results = runner.get_summaries()
    runner.reset_summaries()
    logger.info("*** evaluation log ***")
    log_results(runner.t_env, eval_results, logger, mode="eval", episodes=eval_episodes,
                log_video=False, log_agent_returns=log_agent_returns)

    # save 
    if save_dir is not None:
        # with open(os.path.join(save_dir, "eval_rollouts.pkl"), "w") as f:
        #     pickle.dump(eval_episodes, f)
        with open(os.path.join(save_dir, "eval_results.pkl"), "wb") as f:
            pickle.dump(eval_results, f)

        ifi = 1 / fps  # inter-frame interval
        if save_gifs:
            if save_gifs_num < 0:
                gif_num = n_episodes
            else:
                gif_num = min(save_gifs_num, n_episodes)
            # stack all episodes frames to 1 video sequence
            frames = [
                eval_episodes["frame"][i,j].cpu().numpy()
                for i in range(gif_num) 
                for j in range(eval_episodes.max_seq_length)
            ]
            h, w, c = frames[0].shape
            stacked_frames = np.stack(frames,0).astype(np.uint8).reshape(-1,h,w,c)
            logger.log_video("eval_frames.gif", stacked_frames)
            # imageio.mimsave(os.path.join(save_dir, "eval_frames.gif"),
            #                 frames, duration=ifi)
    return eval_episodes, eval_results

###############################################


    
#####################################################################################
### main
####################################################################################

def run(args):
    """ main entry func """
    # NOTE: evaluation setup
    config = setup_evaluation(args)
    logger = ExperimentLogger(config.save_dir, log_std_out=True, use_tensorboard=False)

    # NOTE: specify `config.checkpoint` or `config.restore_model`
    if config.checkpoint > 0:
        model_path = "checkpoints/model_{}.ckpt".format(config.checkpoint)
    else:
        model_path = "model.ckpt"
    model_path = os.path.join(config.restore, model_path)
    # model_path = config.restore_model
    # load agent 
    learner = RMADDPG.init_from_save(model_path)
    if config.copy_checkpoint:
        learner.save(config.save_dir + "/model.ckpt")

    # NOTE: make sampling env runner 
    env_func = ENV_MAP[config.env]
    p_env_func = partial(env_func, config.scenario, benchmark=False, 
                        show_visual_range=config.show_visual_range)
    env = make_parallel_env(p_env_func, config.env_config, config.eval_batch_size, 
                            config.n_rollout_threads, config.seed)

    #################################################### eval 
    # # NOTE: evaluate with direct rollouts 
    # rollouts = maddpg_rollout(
    #     maddpg, env, config.n_episodes, config.episode_length, 
    #     logger=logger, render=True, save_gifs=True, fps=20
    # )

    # NOTE: evaluate with runner 
    scheme = get_sample_scheme(learner.nagents, env.observation_space, env.action_space)
    runner = EpisodeRunner(scheme, env, learner, logger, config.eval_batch_size,
                            config.episode_length, device=config.device, t_env=0, 
                            is_training=False)

    # save rollout trajectories, benchmark data and video 
    rollouts, results = maddpg_rollout_runner(
        config, learner, runner, config.n_episodes, config.episode_length, 
        logger=logger, render=True, save_dir=config.save_dir, fps=20, 
        save_gifs=True, save_gifs_num=config.save_gifs_num
    )
    
    # clean up
    env.close()
    logger.info("Finished Evaluation")
    logger.close()



if __name__ == '__main__':
    args = parse_args()
    run(args)