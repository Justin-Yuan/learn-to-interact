import os 
import time
import pickle 
import imageio
import argparse
import numpy as np
from pathlib import Path
import torch
from torch.autograd import Variable

from runners.make_env import ENV_MAP
from algorithms.maddpg import MADDPG
from utils.exp_utils import setup_evaluation, ExperimentLogger 


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
    parser.add_argument("--checkpoint", default=-1, type=int,
                        help="Load incremental policy from given episode")
    parser.add_argument("--copy_checkpoint", default=False, action="store_true",
                        help="if to copy the evaluated checkpoint to current eval directory")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--save_gifs", default=False, action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--save_gifs_num", default=-1, type=int,
                        help="number of episode gifs to save")
    parser.add_argument("--fps", default=30, type=int, 
                        help="frame per second of generated gif")

    # Environment
    parser.add_argument("--env", type=str, default="mpe-hier",
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

def maddpg_rollouts(maddpg, env, n_episodes=10, episode_length=25, logger=None, 
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

    return rollouts 

    
#####################################################################################
### main
####################################################################################

def run(config):
    """ main entry func """
    config = setup_evaluation(args)
    logger = ExperimentLogger(config.save_dir, log_std_out=True, use_tensorboard=False)

    # load agent from checkpoint 
    if config.checkpoint > 0:
        model_path = "checkpoints/model_ep{}.ckpt".format(config.checkpoint)
    else:
        model_path = "model.ckpt"
    model_path = os.path.join(config.restore, model_path)
    maddpg = MADDPG.init_from_save(model_path)
    if config.copy_checkpoint:
        maddpg.save(config.save_dir + "/model.ckpt")

    # make env runner 
    env_func = ENV_MAP[config.env]
    env = env_func(config.scenario, benchmark=False, 
                        show_visual_range=config.show_visual_range, **config.env_config)
    
    # evaluate 
    rollouts = maddpg_rollouts(maddpg, env, config.n_episodes, config.episode_length, 
                    logger=logger, render=True, save_gifs=True, fps=20)
    
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

    env.close()


if __name__ == '__main__':
    args = parse_args()
    run(args)