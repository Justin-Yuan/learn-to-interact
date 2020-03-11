import os 
import sys 
import shutil 
import argparse 
import json 
import yaml 
import shlex 
import subprocess
import matplotlib.pyplot as plt 
import seaborn as sns 

#####################################################################################
### eval funcs 
####################################################################################

def parse_args():
    parser = argparse.ArgumentParser("MADDPG with OpenAI MPE")
    parser.add_argument("-m", "--mode", type=str, default="remove",
                        help="mode of clean up")
    parser.add_argument("-r", "--root_dir", type=str, default="cluster_exps/exps", help="root directory for batch evaluation")
    return parser.parse_args()
    

#####################################################################################
### eval funcs 
####################################################################################

def run_rollouts(root_dir):
    """ generatee rollouts for all experiemnts under root_dir 
    """
    # find all paths to exp-level directories
    exp_paths = [
        x[0] for x in os.walk(root_dir)
        if "seed" in x[0].split("/")[-1]
    ]   
    for path in exp_paths:
        print("Evaluating: ", path)
        # construct eval command
        command = "python algorithms/rmaddpg/eval_rmaddpg.py --save_gifs --use_restore_env_config --use_restore_config --restore {}".format(path)
        # exec evaluation 
        try:
            subprocess.call(command, shell=True)
        except:
            print("Fail to evaluate: ", path)


def plot_returns(root_dir):
    """ generate returns plots over several algos (and seeds) 
        with std and legends
    """
    plt.figure()




#####################################################################################
### eval funcs 
####################################################################################

if __name__ == '__main__':
    args = parse_args()
    run_rollouts(args.root_dir)