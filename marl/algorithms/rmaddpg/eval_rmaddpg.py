import os 
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
from algorithms.maddpg import RMADDPG
from runners.sample_batch import EpisodeBatch
from runners.episode_runner import EpisodeRunner
from utils.exp_utils import setup_evaluation, ExperimentLogger 
from utils.exp_utils import time_left, time_str, merge_dict

from .run_rmaddpg import get_sample_scheme, make_parallel_env, log_results


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

    # evaluation
    parser.add_argument("--restore", type=str, default=None,
                        help="directory in which training state and model are loaded")
    parser.add_argument("--restore_model", type=str, default=None,
                        help="file for the checkpoint file to load")
    parser.add_argument("--checkpoint", default=-1, type=int,
                        help="Load incremental policy from given episode (alternative to restore_model)")
    parser.add_argument("--copy_checkpoint", default=False, action="store_true",
                        help="if to copy the evaluated checkpoint to current eval directory")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--sample_batch_size", default=1, type=int,
                        help="number of data points sampled () per run, use 1 for evaluation")
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
    parser.add_argument("--env_config", type=str, default="",
                        help="file to environment scenario config")
    parser.add_argument("--use_restore_env_config", default=False, action='store_true',
                        help="if to use env config from retore dierctory")
    parser.add_argument("--episode_length", default=25, type=int,
                        help="max episode length")
    parser.add_argument("--show_visual_range", default=False, action='store_true', 
                        help='if to show agent visual range when rendering')

    args = parser.parse_args()
    return args


#####################################################################################
### funcs 
####################################################################################

def maddpg_rollout(maddpg, env, n_episodes=10, episode_length=25, logger=None, 
                    render=False, save_gifs=False, fps=20, **kwargs):
    """ get evaluation rollouts, return rollouts
    """
    rollouts = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "next_obs": [],
        "episode_ids": []
    }
    maddpg.prep_rollouts(device="cpu")
    ifi = 1 / fps  # inter-frame interval

    for ep_i in range(n_episodes):
        if logger is not None: 
            logger.info("Episode %i/%i" % (ep_i + 1, n_episodes))
        episode_rewards = []
        
        # init episode
        obs = env.reset()
        if save_gifs:
            rollouts["frames"] = []
            rollouts["frames"].append(env.render('rgb_array')[0])
        if render:
            env.render('human')

        for t_i in range(episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]

            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]

            # step env 
            next_obs, rewards, dones, infos = env.step(actions)
            rollouts["obs"].append(obs)
            rollouts["actions"].append(actions)
            rollouts["rewards"].append(rewards)
            rollouts["dones"].append(dones)
            rollouts["next_obs"].append(next_obs)
            rollouts["episode_ids"].append(ep_i)
            obs = next_obs
            episode_rewards.append(rewards)

            # render 
            if save_gifs:
                rollouts["frames"].append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            if render:
                env.render('human')

        # episodic summary 
        if logger is not None:
            # rewards (T,N) -> (N,T)
            agent_rewards = list(zip(*episode_rewards))
            agent_mean_rewards = [np.mean(ar) for ar in agent_rewards]
            episode_mean_reward = np.mean(agent_rewards)
            n_agents = len(agent_rewards)
            
            log_str = "eps_mean_rew: {}".format(episode_mean_reward)
            log_str += " | " + " ".join([
                "agent_{}: {}".format(i, agent_mean_rewards[i]) 
                for i in range(n_agents)
            ])
            logger.info(log_str)

    # save rollouts 
    if save_dir is not None:
        with open(os.path.join(save_dir, "eval_rollouts.pkl"), "w") as f:
            pickle.dump(rollouts, f)

        if config.save_gifs:
            if config.save_gifs_num < 0:
                gif_num = config.n_episodes
            else:
                gif_num = min(config.save_gifs_num, config.n_episodes)
            imageio.mimsave(os.path.join(save_dir, "eval_frames.gif"),
                            rollouts["frames"][:gif_num], duration=ifi)

    return rollouts 

###############################################

def maddpg_rollout_runner(maddpg, runner, n_episodes=10, episode_length=25, logger=None, 
                    render=False, save_dir=None, fps=20, save_gifs=False, save_gifs_num=-1, **kwargs):
    """ get rollouts with runner 
    """
    eval_rollouts = []
    eval_results = []

    n_test_runs = max(1, n_episodes // runner.batch_size)
    assert n_episodes // runner.batch_size == 0, "n_episodes should be divisible by batch_size"
    maddpg.prep_rollouts(device=config.device)

    for _ in range(n_test_runs):
        eval_batch, eval_res = runner.run(test_mode=True, render=render)
        eval_rollouts = append(eval_batch)
        eval_results.append(eval_res)

    # collect evaluation stats
    eval_rollouts = EpisodeBatch.concat(eval_rollouts)
    eval_results = merge_dict(*eval_results)
    logger.info("*** evaluation log ***")
    log_results(t_env, eval_results, logger, mode="eval")

    # save 
    if save_dir is not None:
        with open(os.path.join(save_dir, "eval_rollouts.pkl"), "w") as f:
            pickle.dump(eval_rollouts, f)
        with open(os.path.join(save_dir, "eval_results.pkl"), "w") as f:
            pickle.dump(eval_results, f)

        ifi = 1 / fps  # inter-frame interval
        if save_gifs:
            if save_gifs_num < 0:
                gif_num = n_episodes
            else:
                gif_num = min(save_gifs_num, n_episodes)
            frames = [
                eval_rollouts["frame"][i,j].cpu().numpy()
                for i in range(gif_num) for j in range(eval_rollouts.max_sequence_length)
            ]
            imageio.mimsave(os.path.join(save_dir, "eval_frames.gif"),
                            frames, duration=ifi)

    return eval_rollouts, eval_results

    
#####################################################################################
### main
####################################################################################

def run(config):
    """ main entry func """
    # NOTE: evaluation setup
    config = setup_evaluation(args)
    logger = ExperimentLogger(config.save_dir, log_std_out=True, use_tensorboard=False)

    # NOTE: specify `config.checkpoint` or `config.restore_model`
    if config.checkpoint > 0:
        model_path = "checkpoints/model_ep{}.ckpt".format(config.checkpoint)
    else:
        model_path = "model.ckpt"
    model_path = os.path.join(config.restore, model_path)
    # model_path = config.restore_model
    # load agent 
    maddpg = RMADDPG.init_from_save(model_path)
    if config.copy_checkpoint:
        maddpg.save(config.save_dir + "/model.ckpt")

    # NOTE: make sampling env runner 
    env_func = ENV_MAP[config.env]
    p_env_func = partial(env_func, config.scenario, benchmark=False, 
                        show_visual_range=config.show_visual_range)
    env = make_parallel_env(p_env_func, config.env_config, config.sample_batch_size, 
                            config.n_rollout_threads, config.seed)

    #################################################### eval 
    # # NOTE: evaluate with direct rollouts 
    # rollouts = maddpg_rollout(
    #     maddpg, env, config.n_episodes, config.episode_length, 
    #     logger=logger, render=True, save_gifs=True, fps=20
    # )

    # NOTE: evaluate with runner 
    obs_dims = [obsp.shape[0] sfor obsp in env.observation_space]
    act_dims = [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                for acsp in env.action_space]
    scheme = get_sample_scheme(maddpg.nagents, obs_dims, act_dims)
    runner = EpisodeRunner(scheme, env, maddpg, logger, config.sample_batch_size,
                            config.episode_length, device=config.device, render=True)

    rollouts, results = maddpg_rollout_runner(
        maddpg, runner, config.n_episodes, config.episode_length, 
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