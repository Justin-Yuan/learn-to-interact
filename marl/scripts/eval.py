import os 
import sys 
import shutil 
import argparse 
import json 
import yaml 
import numpy as np
import shlex 
import subprocess
import matplotlib.pyplot as plt 
import seaborn as sns 

#####################################################################################
### eval funcs 
####################################################################################

def parse_args():
    parser = argparse.ArgumentParser("MADDPG with OpenAI MPE")
    parser.add_argument("-m", "--mode", type=str, default="run_rollouts",
                        help="mode of clean up")
    parser.add_argument("-r", "--root_dir", type=str, default="cluster_exps/exps", help="root directory for batch evaluation")
    parser.add_argument("-o", "--output", type=str, default="result.png", help="output file/image name")
    parser.add_argument("--keys", type=str, nargs='+',
                        help="keys to include for filtering experiment paths")
    parser.add_argument("--no_keys", type=str, nargs='+',
                        help="keys to exclude for filtering experiment paths")
    parser.add_argument("--last_token", type=str, default="",
                        help="to determine last level of exp path")
    parser.add_argument("--smooth_num", default=1, type=int,
                        help="smooth filter over number of points")
    parser.add_argument("--smooth_ratio", default=0.0, type=float,
                        help="smooth ratio w.r.t sequence length")
    return parser.parse_args()
    

#####################################################################################
### helper funcs 
####################################################################################

colors = [
    "b", # blue
    "g", # green
    "r", # red
    "c", # cyan
    "m", # magenta
    "y", # yellow
    "k", # black
    "w" # white
]

def filter_path(path, keys=None, no_keys=None):
    # if keys to be included is true 
    all_in_true = True 
    if keys is not None:
        all_in_true= all([True if k in path else False for k in keys])
    # if keys to be exluded is true 
    all_out_true = True 
    if no_keys is not None:
        all_out_true = all([False if k in path else True for k in no_keys])
    return all_in_true and all_out_true


def filter_path_by_last(path, last):
    return True if last in path.split("/")[-1] else False


def window_filter(y, smooth=1):
    """ 2k+1 window filter, take weighted average around point for -k and +k steps 
        y: 1D array 
    """
    k = np.ones(smooth)
    y = np.asarray(y)
    z = np.ones(len(y))
    smoothed = np.convolve(y,k,'same') / np.convolve(z,k,'same')
    return smoothed


def iir_filter(y, weight=0.0):
    """ 1st-order IIR low-pass filter to attenuate the higher-frequency components of the time-series
        y: 1D array 
        reference: https://dingguanglei.com/tensorboard-xia-smoothgong-neng-tan-jiu/
        reference: https://blog.csdn.net/Charel_CHEN/article/details/80364841
        reference: https://github.com/tensorflow/tensorboard/blob/f801ebf1f9fbfe2baee1ddd65714d0bccc640fb1/tensorboard/plugins/scalar/vz_line_chart/vz-line-chart.ts
        line 694-708
    """
    last = y[0]
    smoothed = []
    for v in y:
        smoothed_v = last * weight + (1 - weight) * v
        smoothed.append(smoothed_v)
        last = smoothed_v
    return np.asarray(smoothed)


#####################################################################################
### eval funcs 
####################################################################################

def run_rollouts(args):
    """ generatee rollouts for all experiemnts under root_dir 
    """
    # find all paths to exp-level directories
    exp_paths = [
        x[0] for x in os.walk(args.root_dir)
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


def plot_returns(args):
    """ generate returns plots over several algos (and seeds) 
        with std and legends
    # NOTE: all seed runs need to have same x data
    """
    # find valid experiment paths
    exp_paths = [
        x[0] for x in os.walk(args.root_dir)
        if filter_path(x[0], args.keys, args.no_keys) and filter_path_by_last(x[0], args.last_token)
    ] 
    assert len(exp_paths) == 1
    exp_path = exp_paths[0]
    # get algo paths 
    algo_paths = [
        os.path.join(exp_path, p) for p in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, p))
    ]
    # collect results 
    data = {} 
    for algo_path in algo_paths:
        algo_name = algo_path.split("/")[-1]
        data[algo_name] = {"xdata": [], "ydata": []}
        # different seeding runs 
        seed_paths = [
            os.path.join(algo_path, p) for p in os.listdir(algo_path) if os.path.isdir(os.path.join(algo_path, p))
        ]
        for seed_path in seed_paths:
            returns_path = os.path.join(seed_path, "eval/returns_mean.log")
            # read returns file (comma separated, per-line)
            try:
                xs, ys = [], []
                with open(returns_path, "r") as f:
                    for l in f.readlines():
                        items = [it.strip() for it in l.strip().split(",")]   # remove trailing newline
                        if len(items) > 0:
                            xs.append(int(items[0]))
                            ys.append(float(items[1]))
                data[algo_name]["xdata"].append(xs)
                data[algo_name]["ydata"].append(ys)
            except:
                print("Failed to read returns from: ", returns_path)
    # postprocess results
    for algo, adata in data.items():
        # truncate to shortest y sequence
        min_len = min([len(y) for y in adata["ydata"]])
        adata["xdata"] = [x[:min_len] for x in adata["xdata"]]
        adata["ydata"] = [y[:min_len] for y in adata["ydata"]]
        # smooth 
        smooth = 1
        if args.smooth_ratio > 0:
            smooth = int(min_len * args.smooth_ratio)
        elif args.smooth_num > 1:
            smooth = args.smooth_num
        if smooth > 1:
            k = np.ones(smooth)
            for i in range(len(adata["ydata"])):
                adata["ydata"][i] = window_filter(adata["ydata"][i], smooth=smooth)
                # adata["ydata"][i] = iir_filter(adata["ydata"][i], weight=args.smooth_ratio)
    # start plotting 
    plt.figure()
    sns.set(style="darkgrid")
    for i, (algo, adata) in enumerate(data.items()):
        x = adata["xdata"][0]
        y = adata["ydata"]
        c = colors[i % len(colors)]
        sns.tsplot(time=x, data=y, color=c, condition=algo)
    
    plt.title("Eval mean returns", fontsize=20)
    plt.xlabel("steps", fontsize=15)#, labelpad=-4)
    plt.ylabel("returns", fontsize=15)#, labelpad=-4)
    # plt.legend(loc="lower right")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout(pad=0.5)

    # save plot  
    plt.savefig(args.output)
    print("Saved output: ", args.output)



#####################################################################################
### eval funcs 
####################################################################################

if __name__ == '__main__':
    args = parse_args()
    if args.mode == "run_rollouts":
        run_rollouts(args)
    elif args.mode == "plot_returns":
        plot_returns(args)
    else:
        print("Unknown evaluation mode...")