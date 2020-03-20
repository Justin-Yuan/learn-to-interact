# MARL 

training commands 

```bash 
python main.py 
```


submodules 
reference: https://www.vogella.com/tutorials/GitSubmodules/article.html

add submodule to repo 
```bash
git submodule add -b master URL
git submodule init 
```

- to clone with submodules 
```bash 
git clone --recursive URL 
```

- to pull updates from `mpe_hierarchy` submodule, do 
```bash
git submodule update --remote
```

## Notes 

- runner scheme: make_env -> wrap in vec_env -> build runner 



## Pip install 
```bash
cat requirements.txt | xargs -n 1 pip install
```

 Xvfb -screen 0 1024x768x24 & 



import gym, PIL
env = gym.make('SpaceInvaders-v0')
array = env.reset()
PIL.Image.fromarray(env.render(mode='rgb_array'))


## debugging simple_speaker_listener and simple_tag 
- norm rewards not the cause,
    all speaker-listener, tag and spread works well 
- norm inputs not the cause 
    speaker-listener and tag works well 
- norm both not the cause 
    speaker-listenere and tag works well without both