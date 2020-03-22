import os 
import sys 
import yaml 
import json 
import time 
import errno
import pickle 
import random
import logging 
import imageio
import numpy as np 
import termcolor as tc
from datetime import datetime 
from dict_deep import deep_get, deep_set, deep_del

import torch
import torch.nn as nn
import torch.nn.functional as F 
from tensorboardX import SummaryWriter

import utils.bunch as bunch

#####################################################################################
### arguments 
#####################################################################################

def mkdirs(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def eval_token(token):
    """ convert string token to int, float or str """ 
    if token.isnumeric():
        return int(token)
    try:
        return float(token)
    except:
        return token


def read_file(file_path, sep=","):
    """ read a file (json, yaml, csv, txt)
    """
    if len(file_path) < 1 or not os.path.exists(file_path):
        return None 
    # load file 
    f = open(file_path, "r")
    if "json" in file_path:
        data = json.load(f)
    elif "yaml" in file_path:
        data = yaml.load(f)
    else:
        sep = sep if "csv" in file_path else " "
        data = []
        for line in f.readlines():
            line_post = [eval_token(t) for t in line.strip().split(sep)]
            # if only sinlge item in line 
            if len(line_post) == 1:
                line_post = line_post[0]
            if len(line_post) > 0:
                data.append(line_post)
    f.close()
    return data


def overwrite_config(line, config):
    """ args.overwrite has a string to overwrite configs 
    overwrite string format (flat hierarchy): 'nested_name-nested_type-value; ...' 
    nested_name: a.b.c; nested_type: type or list.type 
    """
    if line is None or len(line) == 0:
        return config 
    # parse overwrite string
    for item in line:
        cname, ctype, cval = [e.strip() for e in item.strip().split("-")]
        types = ctype.split(".")
        base_type = types[-1]
        # make value 
        if len(types) > 1:  # list of basic types 
            val = cval[1:-1].split(",")    # exclude brackets []
            if base_type == "bool":
                val = [False if e.strip() == "false" else True for e in val] 
            else:
                val = [eval(base_type)(e.strip()) for e in val]
            val = eval(types[0])(val)   # only 2 levels allowed
        else:   # int, float, bool, str
            if base_type == "bool": 
                val = False if cval == "false" else True
            else:
                val = eval(base_type)(cval)
        # update config 
        deep_set(config, cname, val)
    return config 


def set_seed(seed, cuda=False):
    """ for reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


#####################################################################################
### experiment management 
#####################################################################################

class ExperimentState():
    """ minimal set of tracking stats to restore halted experiment 
    """
    state_fields = [
        "episode", 
        "t_env", 
        "last_log_t", 
        "last_train_t",
        "last_target_update_t", 
        "last_train_log_t", 
        "last_test_t", 
        "last_save_t"
    ]
    def __init__(self):
        self.reset()

    def reset(self):
        """ do match with `state_fields` """
        self.episode = 0     # total episodes so far
        self.t_env = 0       # total env interacetion steps so far
        self.last_log_t = 0
        self.last_train_t = 0
        self.last_target_update_t = 0
        self.last_train_log_t = 0
        self.last_test_t = 0
        self.last_save_t = 0

    def save_state(self, f_name):
        """  convenient for restoring halted experient """
        try:
            state = {
                k: self.__dict__[k] 
                for k in ExperimentState.state_fields
            }
            with open(f_name, "wb") as f:
                pickle.dump(state, f) 
        except:
            print("Save experiment state failed...")

    def load_state(self, f_name):
        """ load exp state to restore halted experiment """
        try:
            with open(f_name, "rb") as f:
                state = pickle.load(f)
            for k, v in state.items():
                self.__dict__[k] = v            
        except:
            print("Load experiment state failed...")


############################################ train     
def setup_experiment(args):
    """ general setup and book keeping 
    """
    # restore halted experiment, otherwise start anews
    if args.restore is not None and os.path.exists(args.restore):
        return restore_experiment(args)

    # experiment book-keeping 
    exp_dir = [args.save_dir]

    temp = []   # top-level dir

    for e in ["exp", "env", "scenario"]:    
        if hasattr(args, e):
            temp.append(getattr(args, e).replace("_", "-"))
    exp_dir.append("_".join(temp))

    if args.sub_dir is not None and len(args.sub_dir) > 0:
        exp_dir += args.sub_dir
    if args.tag is not None and len(args.tag) > 0:
        exp_dir.append("_".join(args.tag))

    timestamp = datetime.now().strftime("%b-%d-%H-%M")  # till minute %Y-%b-%d-%H-%M
    run_dir = "seed{}_{}_{}".format(str(args.seed), str(timestamp), str(os.getpid()))
    exp_dir.append(run_dir)

    args.save_dir = os.path.join(*exp_dir)   # save_dir/exp_env/*sub_dir/tag/run#0
    mkdirs(args.save_dir)

    # env configs 
    env_config = read_file(args.env_config)
    env_config = {} if env_config is None else env_config
    if args.overwrite is not None and len(args.overwrite) > 0:
        env_config = overwrite_config(args.overwrite, env_config)

    # snapshot configs 
    args_dict = vars(args)
    with open(args.save_dir+"/config.yaml", "w") as f:
        yaml.dump(args_dict, f, default_flow_style=False)
    with open(args.save_dir+"/env_config.yaml", "w") as f:
        yaml.dump(env_config, f, default_flow_style=False)

    # combine all configs to 1 namespace
    args_dict["env_config"] = env_config 
    config = bunch.bunchify(args_dict)

    # set device 
    use_cuda = config.cuda and torch.cuda.is_available()
    if not use_cuda:
        torch.set_num_threads(config.n_training_threads)
    config.device = "cuda" if use_cuda else "cpu"
    # config.device = torch.device("cuda" if use_cuda else "cpu")

    # set seed 
    seed = config.seed 
    if seed > 0:
        set_seed(seed, cuda=config.cuda)

    return config, False


############################################ restore
def restore_experiment(args):
    """ restore from halted experiment, load exp state, config, 
    models from `restore` (save_dir/exp_env/*sub_dir/tag/run#0),
    higher date correspond to more recent experiment of same seeds
    """
    # make resume experiment in new directory
    exp_dir = [os.path.dirname(args.restore)] 
    timestamp = datetime.now().strftime("%b-%d-%H-%M")  # till minute
    run_dir = "seed{}_{}_{}".format(str(args.seed), str(timestamp), str(os.getpid()))
    exp_dir.append(run_dir)

    save_dir = os.path.join(*exp_dir)   # save_dir/exp_env/*sub_dir/tag/run#0
    mkdirs(save_dir)

    # retrieve configs 
    args_dict = read_file(args.restore+"/config.yaml")
    args_dict["restore"] = args.restore     # maintain `restore` filed
    args_dict["save_dir"] = save_dir    # update new save_dir
    env_config = read_file(args.restore+"/env_config.yaml")

    # snapshot to current save_dir
    with open(save_dir+"/config.yaml", "w") as f:
        yaml.dump(args_dict, f, default_flow_style=False)
    with open(save_dir+"/env_config.yaml", "w") as f:
        yaml.dump(env_config, f, default_flow_style=False)

    # combine all configs to 1 namespace (default)
    args_dict["env_config"] = env_config 
    config = bunch.bunchify(args_dict)
    config.restore_model = args.restore + "/model.ckpt"
    config.restore_exp_state = args.restore +  "/exp_state.pkl"

    # specific overwrites (e.g. longer episodes or total steps)
    if args.restore_model is not None and os.path.exists(args.restore_model):
        config.restore_model = args.restore + "/model.ckpt"
    if args.n_episodes > config.n_episodes:
        config.n_episodes = args.n_episodes
    if args.n_env_steps > config.n_env_steps:
        config.n_env_steps = args.n_env_steps
    
    # set device 
    use_cuda = config.cuda and torch.cuda.is_available()
    if not use_cuda:
        torch.set_num_threads(config.n_training_threads)
    config.device = "cuda" if use_cuda else "cpu"
    # config.device = torch.device("cuda" if use_cuda else "cpu")

    # set seed 
    seed = config.seed 
    if seed > 0:
        set_seed(seed, cuda=config.cuda)

    return config, True


############################################ eval 
def setup_evaluation(args):
    """ for general evaluation
    """
    assert args.restore is not None and os.path.exists(args.restore)

    # if to use original exp config 
    if args.use_restore_config:
        orig_config = read_file(args.restore + "/config.yaml")
        for k in ["exp", "sub_dir", "tag", "seed", "env", "scenario"]:
            setattr(args, k, orig_config[k])
        # other optional fields 
        for k in ["ensemble_size", "ensemble_config"]:
            if k in orig_config:
                setattr(args, k, orig_config[k])

    exp_dir = [args.save_dir]
    temp = []   # top-level dir

    for e in ["exp", "env", "scenario"]:    
        if hasattr(args, e):
            temp.append(getattr(args, e).replace("_", "-"))
    exp_dir.append("_".join(temp))

    if args.sub_dir is not None and len(args.sub_dir) > 0:
        exp_dir += args.sub_dir
    if args.tag is not None and len(args.tag) > 0:
        exp_dir.append("_".join(args.tag))

    timestamp = datetime.now().strftime("%b-%d-%H-%M")  # till minute
    run_dir = "seed{}_{}_{}".format(str(args.seed), str(timestamp), str(os.getpid()))
    exp_dir.append(run_dir)

    args.save_dir = os.path.join(*exp_dir)   # save_dir/exp_env/*sub_dir/tag/run#0
    mkdirs(args.save_dir)

    # restore 
    restore_dir = args.restore
    if args.use_restore_env_config or args.env_config is None:
        env_config = read_file(args.restore + "/env_config.yaml")
    else:
        env_config = read_file(args.env_config)
    
    # config snapshot 
    args_dict = vars(args)
    with open(args.save_dir+"/config.yaml", "w") as f:
        yaml.dump(args_dict, f, default_flow_style=False)
    with open(args.save_dir+"/env_config.yaml", "w") as f:
        yaml.dump(env_config, f, default_flow_style=False)
    args_dict["env_config"] = env_config 
    config = bunch.bunchify(args_dict)

    # set checkpoint path
    if config.restore_model is None:
        config.restore_model = args.restore + "/model.ckpt"

    # set device 
    use_cuda = config.cuda and torch.cuda.is_available()
    config.device = "cuda" if use_cuda else "cpu"

    # set seed 
    seed = config.seed 
    if seed > 0:
        set_seed(seed, cuda=False)

    return config 


#####################################################################################
### logging  
#####################################################################################

def set_print_logger(logger_name, log_file, level=logging.INFO):
    """ print logger to std_out and log file
    """
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    # log to file 
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    # log to std out 
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    return logger 


class Logger(object):
    """
    Base Logger object

    Initializes the log directory and creates log files given by name in arguments.
    Can be used to append future log values to each file.
    """
    def __init__(self, log_dir, *args):
        self.log_dir = log_dir

        try:
            os.makedirs(log_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        with open(os.path.join(self.log_dir, 'cmd.txt'), 'w') as f:
            f.write(" ".join(sys.argv))

        self.log_names = [a for a in args]
        for arg in self.log_names:
            setattr(self, 'log_{}'.format(arg), lambda epoch, value, name=arg: self.log(name, epoch, value))
            self.init_logfile(arg)

    def log_config(self, config):
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

    def init_logfile(self, name, xlabel="epoch"):
        fname = self.get_log_fname(name)
        with open(fname, 'w') as log_file:
            log_file.write("{},{}\n".format(xlabel, name))

    def get_log_fname(self, name):
        return os.path.join(self.log_dir, '{}.log'.format(name))

    def log(self, name, value, epoch):
        if name not in self.log_names:
            try:  # initialize only if not done so already
                self.get_log_fname(name)
            except FileNotFoundError:
                self.init_logfile(name)
            self.log_names.append(name)
        fname = self.get_log_fname(name)
        # TODO: dirty hack
        dir_name = os.path.dirname(os.path.realpath(fname))
        mkdirs(dir_name)

        with open(fname, 'a') as log_file:
            log_file.write("{},{}\n".format(epoch, value))

    def log_test_value(self, name, value):
        test_name = 'test_' + name
        self.init_logfile(test_name)
        self.log(test_name, 0, value)


class ExperimentLogger(object):
    """Wraps Tensorboard logger and lightweight logger. (and standard output logger)
    TensorBoard logger can be useful for looking at all experiments

    Performs checkpointing and resumes experiments that are not completed.
    - serves to combine other loggers 
    """
    def __init__(self, log_dir, checkpoint_name="latest.tar", log_std_out=False, use_tensorboard=True):
        self.log_dir = log_dir
        self.logger = Logger(log_dir=log_dir)
        self.checkpoint_name = os.path.join(self.log_dir, checkpoint_name)

        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.tensorboard_writer = SummaryWriter(log_dir=self.log_dir)

        self.log_std_out = log_std_out
        if log_std_out:
            std_out_file = os.path.join(log_dir, "std_out.txt")
            self.std_out_logger = set_print_logger("marl", std_out_file)

    def info(self, msg):
        """ mimic logging.logger """
        if self.log_std_out:
            self.std_out_logger.info(msg)
        else:
            print(msg)

    def update_tensorboard_writer(self, epoch):
        self.tensorboard_writer = SummaryWriter(log_dir=self.log_dir, purge_step=epoch)

    def checkpoint_exists(self):
        return os.path.isfile(self.checkpoint_name)

    def load_checkpoint(self):
        return torch.load(self.checkpoint_name)

    def log_hyperparams(self, hyp_dict, file_name="hyperparams.json"):
        with open(os.path.join(self.log_dir, file_name), 'w') as fp:
            json.dump(hyp_dict, fp)

    def add_scalar(self, name, val, iter):
        self.logger.log(name, val, iter)
        if self.use_tensorboard:
            self.tensorboard_writer.add_scalar(name, val, iter)

    def add_histogram(self, name, val_array, iter):
        self.tensorboard_writer.add_histogram(name, val_array, iter)

    def add_histogram_dict(self, val_dict, iter):
        """ take in dict of named parameters (e.g. network weights) """
        if self.use_tensorboard:
            for k, v in val_dict.items():
                self.add_histogram(k, v, iter)

    def add_images(self, name, image_tensor, iter):
        """ images: (N,C,H,W) """ 
        self.tensorboard_writer.add_images(name, image_tensor, iter)

    def add_video(self, name, video_tensor, iter):
        """ videos: (T,H,W,C) """
        self.tensorboard_writer.add_video(name, video_tensor, iter, fps=4)

    def log_video(self, name, video, fps=20):
        """ video: rgb arrays 
        reference: https://imageio.readthedocs.io/en/stable/format_gif-pil.html
        """
        vid_kargs = {
            'fps': fps   # duration per frame
        }   
        vid_name = '{}/{}'.format(self.log_dir, name)
        mkdirs(os.path.dirname(vid_name))   # often is "videos/"
        imageio.mimsave(vid_name, video, **vid_kargs)

    def log_epoch(self, epoch, state, epoch_stats):
        assert 'epoch' in state
        assert 'model' in state
        assert 'optimizer' in state

        torch.save(state, self.checkpoint_name)
        for k, v in epoch_stats.items():
            self.add_scalar(k, v, epoch)
    
    def export_scalars_to_json(self, summary_path="summary.json"):
        self.tensorboard_writer.export_scalars_to_json(self.log_dir + "/" + summary_path)

    def close(self):
        if self.use_tensorboard:
            self.tensorboard_writer.close()



#####################################################################################
### timing  
#####################################################################################

def time_left(start_time, t_start, t_current, t_max):
    if t_current >= t_max:
        return "-"
    time_elapsed = time.time() - start_time
    t_current = max(1, t_current)
    time_left = time_elapsed * (t_max - t_current) / (t_current - t_start)
    # Just in case its over 100 days
    time_left = min(time_left, 60 * 60 * 24 * 100)
    return time_str(time_left)


def time_str(s):
    """
    Convert seconds to a nicer string showing days, hours, minutes and seconds
    """
    days, remainder = divmod(s, 60 * 60 * 24)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    string = ""
    if days > 0:
        string += "{:d} days, ".format(int(days))
    if hours > 0:
        string += "{:d} hours, ".format(int(hours))
    if minutes > 0:
        string += "{:d} minutes, ".format(int(minutes))
    string += "{:d} seconds".format(int(seconds))
    return string



#####################################################################################
### misc  
#####################################################################################

def merge_dict(*dicts):
    """ combine list of dicts, add new keys or concat on same keys 
        assume dict only union keys only contain only 
        int/float/list/np.array/torch.tensor/dict
    """
    res = {}
    for d in dicts:
        union_d = {k:v for k, v in d.items() if k in res}
        exclusion_d = {k:v for k, v in d.items() if k not in res}
        res.update(exclusion_d)
        for k, v in union_d.items():
            if isinstance(v, (int, float)):
                res[k] += v
            elif isinstance(v, (list, tuple)):
                res[k] += v 
            elif isinstance(v, np.array):
                if len(v.shape) == 0 or v.shape[0] == 1:
                    # if scalar or global to batch, interpret as addition
                    res[k] += v
                else:
                    # other wise interpret as concatenation along 1st dim
                    res[k] = np.concatenate([res[k], v], 0)
            elif isinstance(v, torch.Tensor):
                if len(v.shape) == 0 or v.shape[0] == 1:
                    res[k] += v
                else:
                    res[k] = torch.concat([res[k], v], 0)
            elif isinstance(v, dict):
                res[k] = merge_dict(res[k], v)
    return res


#####################################################################################
### main/test
#####################################################################################

def test_merge_dict():
    pass 





####################################
if __name__ == "__main__":
    pass 