"""
Modified from OpenAI Baselines code to work with multi-agent envs
reference: https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
"""
import numpy as np
from multiprocessing import Process, Pipe

from runners.vec_env import VecEnv
        

#####################################################################################
### funcs
####################################################################################

def worker(remote, parent_remote, env_fn_wrapper):
    """ worker func to execute vec_env commands 
    """
    def step_env(env, action):
        ob, reward, done, info = env.step(action)
        if done:
            ob = env.reset()
        return ob, reward, done, info

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            # branch out for requests 
            if cmd == 'step':
                res = [step_env(env, action) for env, action in zip(envs, data)]
                remote.send(res)
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'render':
                remote.send([env.render(mode='rgb_array') for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send(CloudpickleWrapper(
                    (envs[0].observation_space, envs[0].action_space)
                ))
            elif cmd == 'get_agent_types':
                if all([hasattr(a, 'adversary') for a in envs[0].agents]):
                    res = [
                        'adversary' if a.adversary else 'agent' 
                        for a in envs[0].agents
                    ]
                else:   # fully cooperative
                    res = ['agent' for _ in envs[0].agents]
                remote.send(res)
            else:
                raise NotImplementedErrors
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    except:
        print('Environment runner process failed...')
    finally:
        for env in envs:
            env.close()
            

######################################## misc 
def _flatten_obs(obs):
    
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    # if isinstance(obs[0], dict):
    #     # dict, {key: (B,*key_obs_shape_per_env), ...}
    #     keys = obs[0].keys()
    #     # dict of (B,N,D)
    #     return {k: np.stack([o[k] for o in obs]) for k in keys}
    # else:
    #     # (B,N,O), B = #workers * #envs_per_worker -> (B,N,O)
    #     return np.stack(obs)
    if isinstance(obs[0], dict):
        # dict, {key: (B,*key_obs_shape_per_env), ...}
        keys = obs[0].keys()
        # dict of (B,N,D)
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        # (B,N,O), B = #workers * #envs_per_worker -> (B,N,O)
        return np.stack(obs)


def _flatten_list(l):
    """ convert multiple remotes of obs (each from multiple envs) to 1 list of obs
    """
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


#####################################################################################
### multiprocess envs 
####################################################################################

class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None, n_workers=1):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False

        self.nenvs = len(env_fns)
        self.n_workers = n_workers
        assert nenvs % n_workers == 0, "Number of envs must be divisible by number of workers to run in series"

        env_fns = np.array_split(env_fns, self.n_workers)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_workers)])
        self.ps = [
            Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv().x
        # agent algo types: [agent, adversary, ...]
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        self.viewer = None 
        
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.n_workers)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return  _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    def get_images(self):
        """ called by parent `render` to support tiling images """
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = _flatten_list(imgs)
        return imgs

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"s


#####################################################################################
### single thread env (allow multiple envs sequentially)
####################################################################################

class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.nenvs = len(self.envs)     
        env = self.envs[0]  

        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]

        self.ts = np.zeros(len(self.envs), dtype='int') 
        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]       
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos =zip(*results)
        self.ts += 1
        for (i, done) in enumerate(dones):
            if all(done): 
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        self.actions = None
        return _flatten_obs(obs), np.array(rews), np.array(dones), infos

    def reset(self):        
        obs = [env.reset() for env in self.envs]
        return _flatten_obs(obs)

    def close(self):
        return

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)